// Illustrative, explicitly called instrumentation. Not the Python data source.
// Compile with UVM before rtl/quality_monitor_smoke.sv. No DUT is required.
`include "uvm_macros.svh"
package quality_runtime_metrics_pkg;
  timeunit 1ns;
  timeprecision 1ps;
  import uvm_pkg::*;

  class quality_monitor extends uvm_component;
    `uvm_component_utils(quality_monitor)
    localparam int unsigned TOTAL_BINS = 400;
    typedef bit [31:0] address_t;
    int unsigned transactions, assertion_checks, assertion_failures;
    int unsigned warnings, warning_bursts, errors, protocol_retries;
    int unsigned stimulus_count, corner_cases, randomization_attempts, randomization_passes;
    int unsigned component_messages, initial_bins;
    bit stimulus_seen[int unsigned];
    bit address_seen[address_t];
    bit coverage_seen[int unsigned];
    bit pair_seen[string];
    int unsigned payload_histogram[256];
    realtime started_at, idle_ns, last_warning_at;
    real latency_sum_us;
    bit warning_in_burst;

    function new(string name, uvm_component parent = null);
      super.new(name, parent);
      transactions = 0; assertion_checks = 0; assertion_failures = 0;
      warnings = 0; warning_bursts = 0; errors = 0; protocol_retries = 0;
      stimulus_count = 0; corner_cases = 0;
      randomization_attempts = 0; randomization_passes = 0;
      component_messages = 0; initial_bins = 0;
      started_at = 0.0; idle_ns = 0.0; last_warning_at = 0.0;
      latency_sum_us = 0.0; warning_in_burst = 0;
      foreach (payload_histogram[i]) payload_histogram[i] = 0;
    endfunction

    function void start_of_simulation_phase(uvm_phase phase);
      super.start_of_simulation_phase(phase);
      started_at = $realtime;
    endfunction

    // Call from sequence/driver hooks; reuse below means repeated stimulus types.
    function void record_stimulus(int unsigned kind, bit corner_case = 0);
      stimulus_count++;
      stimulus_seen[kind] = 1;
      if (corner_case) corner_cases++;
    endfunction

    function void record_randomization(bit succeeded);
      randomization_attempts++;
      if (succeeded) randomization_passes++;
    endfunction

    function void record_transaction(address_t address = 0, byte unsigned payload = 0,
                                     real response_latency_us = 0.0);
      if (response_latency_us < 0) begin
        `uvm_error("QUALITY_INPUT", "Response latency must be nonnegative")
        return;
      end
      transactions++;
      address_seen[address] = 1;
      payload_histogram[payload]++;
      latency_sum_us += response_latency_us;
      `uvm_info("QUALITY_METRIC", $sformatf("transaction_count=%0d", transactions), UVM_LOW)
    endfunction

    // Assertion pass/fail action blocks can call this hook; checks are explicit.
    function void record_assertion_check(bit passed = 1);
      assertion_checks++;
      if (!passed) assertion_failures++;
    endfunction

    // An adapter can map a real covergroup's bins to these illustrative bin IDs.
    function void record_coverage_bin(int unsigned bin_id);
      if (bin_id >= TOTAL_BINS) begin
        `uvm_error("QUALITY_INPUT", "Coverage bin ID is outside [0,399]")
        return;
      end
      coverage_seen[bin_id] = 1;
    endfunction

    function void snapshot_initial_coverage();
      initial_bins = coverage_seen.num();
    endfunction

    // Consecutive warnings no more than 10ns apart form one disjoint burst.
    function void record_warning(string detail);
      if ((warnings > 0) && (($realtime - last_warning_at) <= 10ns)) begin
        if (!warning_in_burst) warning_bursts++;
        warning_in_burst = 1;
      end else warning_in_burst = 0;
      warnings++;
      last_warning_at = $realtime;
      `uvm_warning("QUALITY_METRIC", $sformatf("warning_count=%0d detail=%s", warnings, detail))
    endfunction

    function void record_error(string detail);
      errors++;
      `uvm_error("QUALITY_METRIC", detail)
    endfunction

    function void record_retry();
      protocol_retries++;
    endfunction

    // Supply disjoint intervals, e.g. from an interface activity observer.
    function void record_idle(time duration);
      idle_ns += duration;
    endfunction

    function void record_component_message(string sender, string receiver);
      component_messages++;
      pair_seen[{sender, "=>", receiver}] = 1;
    endfunction

    function void report_phase(uvm_phase phase);
      real runtime_ms, entropy, probability, pass_ratio, reuse_ratio, growth, idle_fraction;
      super.report_phase(phase);
      runtime_ms = ($realtime - started_at) / 1ms;
      entropy = 0.0;
      foreach (payload_histogram[i]) begin
        if ((transactions > 0) && (payload_histogram[i] > 0)) begin
          probability = real'(payload_histogram[i]) / transactions;
          entropy -= probability * $ln(probability) / $ln(2.0);
        end
      end
      if (idle_ns > ($realtime - started_at))
        `uvm_error("QUALITY_INPUT", "Recorded idle intervals exceed runtime; check for overlap")
      pass_ratio = assertion_checks ? real'(assertion_checks - assertion_failures) / assertion_checks : 0.0;
      reuse_ratio = stimulus_count ? 1.0 - real'(stimulus_seen.num()) / stimulus_count : 0.0;
      growth = real'(coverage_seen.num() - initial_bins) / TOTAL_BINS;
      idle_fraction = runtime_ms > 0 ? idle_ns / ($realtime - started_at) : 0.0;
      `uvm_info("QUALITY_METRIC", $sformatf("stimulus_types=%0d unique_addresses=%0d payload_entropy=%0f stimulus_reuse_ratio=%0f corner_case_ratio=%0f randomization_success=%0f",
        stimulus_seen.num(), address_seen.num(), entropy, reuse_ratio,
        stimulus_count ? real'(corner_cases) / stimulus_count : 0.0,
        randomization_attempts ? real'(randomization_passes) / randomization_attempts : 0.0), UVM_LOW)
      `uvm_info("QUALITY_METRIC", $sformatf("assertion_checks=%0d assertion_pass_ratio=%0f assertion_failures=%0d coverage_initial=%0f coverage_final=%0f coverage_growth=%0f coverage_bins_hit=%0d",
        assertion_checks, pass_ratio, assertion_failures, real'(initial_bins) / TOTAL_BINS,
        real'(coverage_seen.num()) / TOTAL_BINS, growth, coverage_seen.num()), UVM_LOW)
      `uvm_info("QUALITY_METRIC", $sformatf("warning_count=%0d warning_bursts=%0d error_count=%0d idle_fraction=%0f transaction_count=%0d transactions_per_ms=%0f component_pairs=%0d component_messages=%0d response_latency_us=%0f protocol_retries=%0d runtime_ms=%0f",
        warnings, warning_bursts, errors, idle_fraction, transactions,
        runtime_ms > 0 ? transactions / runtime_ms : 0.0, pair_seen.num(), component_messages,
        transactions ? latency_sum_us / transactions : 0.0, protocol_retries, runtime_ms), UVM_LOW)
    endfunction
  endclass
endpackage
