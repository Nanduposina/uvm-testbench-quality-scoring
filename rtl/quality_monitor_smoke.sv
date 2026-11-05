// Small instrumentation smoke test; no DUT, real design validation or Python feed.
`include "uvm_macros.svh"
package quality_monitor_smoke_pkg;
  timeunit 1ns;
  timeprecision 1ps;
  import uvm_pkg::*;
  import quality_runtime_metrics_pkg::*;

  class quality_monitor_smoke_test extends uvm_test;
    `uvm_component_utils(quality_monitor_smoke_test)
    quality_monitor metrics;

    function new(string name, uvm_component parent = null);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      metrics = quality_monitor::type_id::create("metrics", this);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      metrics.record_coverage_bin(0);
      metrics.snapshot_initial_coverage();
      for (int unsigned i = 0; i < 8; i++) begin
        metrics.record_randomization(1);
        metrics.record_stimulus(i % 3, i == 7);
        metrics.record_transaction(i, byte'(i), 0.1);
        metrics.record_assertion_check(1);
        metrics.record_coverage_bin(i);
        metrics.record_component_message("driver", "monitor");
        metrics.record_idle(10ns);
        #100ns;
      end
      metrics.record_warning("Illustrative warning one");
      #1ns;
      metrics.record_warning("Illustrative warning two, same burst");
      metrics.record_retry();
      #100ns;
      if (metrics.transactions != 8 || metrics.assertion_failures != 0 ||
          metrics.coverage_seen.num() != 8 || metrics.warning_bursts != 1)
        `uvm_fatal("QUALITY_SMOKE", "Instrumentation counters are inconsistent")
      phase.drop_objection(this);
    endtask
  endclass
endpackage

module quality_monitor_smoke;
  timeunit 1ns;
  timeprecision 1ps;
  import uvm_pkg::*;
  import quality_monitor_smoke_pkg::*;
  initial run_test("quality_monitor_smoke_test");
endmodule
