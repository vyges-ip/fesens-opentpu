// The accelerator as built for the PCIe boards: S slices (one on the YPCB-00338, two on the
// BCU1525), each with its DRAM adapter otpu_native_dram onto two memory channels' native ports
// n_* (one command per 64-byte beat, for otpu_mem_ch in front of LiteDRAM's native port: slice
// s on channels 2s and 2s + 1), the collective unit, and the host control registers (AXI4-Lite
// slave). Everything runs on the core clock; the board's top does the clock crossings to the
// memory controllers and the PCIe bridge (otpu_native_sys on the YPCB). The slices share one
// control block: RUN, LOAD (each slice loads the program from its own DRAM) and the arguments
// go to every slice; HALTED when every slice halted, ERROR / WAIT_TO when any did; ICOUNT is
// slice 0's, ICOUNT_s each slice's (0xC0 + 4s); the trace and the activity counters are slice
// 0's, except DRAM_RD / DRAM_WR (every channel) and INSTR (every slice). Slice 0's instances
// keep the one-slice names (u_slice, u_mem), the others are g_slice[s]. The control
// block also holds the free-running activity counters and reads out the hardware trace
// (otpu_trace; docs/observability.md); it also drives the I2C pins low when the host asks and
// reads their levels back (the host bit-bangs I2C).
// At S > 1 every slice sits in its own SLR (the BCU1525's), a crossing away from the control
// block and the collective unit: everything between a slice and them goes through two register
// stages each way (otpu_xr: one on each side of the crossing; g_xs), the slices' control too (so
// they keep starting together), and the collective runs through otpu_coll_hub and a port per
// slice (otpu_coll_port: each slice grants the collective's TMEM accesses on its own). At S = 1
// the signals are wired as before (g_x1: the same logic, tools/synth/board_equiv.sh).
module otpu_board #(
  parameter int S          = 1,         // slices, each on two memory channels
  parameter int D          = 128,
  parameter int MCOLS      = 2,
  parameter int ACT_BLOCKS = 128,
  parameter int ACT_ROWS   = MCOLS,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 15,
  parameter int FIFO_DEPTH = 1024,
  parameter int LANES      = 8,
  parameter int WIN        = 16,
  parameter int RPB        = 8 * LANES,   // every read lane (TMEM is replicated per read port):
                                          // the slice's shallow write-mask arbiter
  parameter int WPB        = 1,
  parameter int MXU_IMPL   = 0,
  parameter int MXU_CL     = 16,
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1,  // VPU lanes with exp2/recip/rsqrt
  parameter int ULANES     = LANES,   // TMEM lanes of the MXU and the quantizer
  parameter int CORE_KHZ    = 100000,  // the core clock (CORE_KHZ register)
  parameter logic [31:0] BUILD_ID = 32'h0,
  parameter int DDR_MTS     = 0,       // DDR3 data rate (DDR_MTS register; 0: not given)
  parameter int TRACE_DEPTH = 16384,   // trace records (a power of two; 0: no trace buffer)
  parameter int TRACE_QD    = 32,      // trace capture queue (cycles with events)
  parameter int PQ_WIN      = 1024,    // cycles per P/Q counter window
  parameter bit HAS_I2C     = 1'b1,    // the I2C pins are wired (CAPS bit2)
  parameter bit CHASH       = 1'b1,    // hashed channel interleave (otpu_native_dram; CAPS bit7)
  parameter bit DSTEP       = 1'b1,    // the DMA's DSTEP datapath (CAPS bit6; 0 leaves it out)
  parameter bit HOSTCAL     = 1'b0,    // the host calibrates the DDR3 controllers (CAPS bit27: LiteDRAM)
  parameter int PLATFORM    = 0,       // the target's platform id (CFG_PLATFORM; opentpu/target.py)
  parameter logic [31:0] ARCH_HASH = 32'h0  // the target's arch_hash (CFG_HASH)
) (
  input  logic         clk,
  input  logic         rst,            // synchronous, active high
  input  logic         ctl_mrst,       // the control master's reset (the SmartConnect's, in clk:
                                       //   PCIe hot reset or link down too): otpu_axil_iso
  input  logic [2*S-1:0] calib,        // memory controllers calibrated (any clock domain)
  input  logic [2*S-1:0] ded,          // a channel's ECC counted an uncorrectable word (any clock
                                       //   domain; STATUS ECC_DED)
  input  logic [11:0]  temp,           // XADC die-temperature code (any clock domain, slow)
  output logic [2:0]   led,
  // ---- I2C pins (otpu_fpga_top_ld's IOBUFs): 1 drives the line low; the levels (any clock)
  output logic [3:0]   i2c_lo,         // SCL0 SDA0 (LM73 bus), SCL1 SDA1 (PCIe SMBus)
  input  logic [4:0]   i2c_pin,        // the same four lines, then ALERT0 (LM73, active low)
  // ---- control: AXI4-Lite slave
  input  logic [11:0]  s_ctl_awaddr,
  input  logic         s_ctl_awvalid,
  output logic         s_ctl_awready,
  input  logic [31:0]  s_ctl_wdata,
  input  logic [3:0]   s_ctl_wstrb,
  input  logic         s_ctl_wvalid,
  output logic         s_ctl_wready,
  output logic [1:0]   s_ctl_bresp,
  output logic         s_ctl_bvalid,
  input  logic         s_ctl_bready,
  input  logic [11:0]  s_ctl_araddr,
  input  logic         s_ctl_arvalid,
  output logic         s_ctl_arready,
  output logic [31:0]  s_ctl_rdata,
  output logic [1:0]   s_ctl_rresp,
  output logic         s_ctl_rvalid,
  input  logic         s_ctl_rready,
  // ---- memory channels 0 .. 2S-1: native masters (otpu_native_dram; slice s on 2s, 2s + 1)
  output logic [2*S-1:0]        n_cvalid,
  input  logic [2*S-1:0]        n_cready,
  output logic [2*S-1:0]        n_cwe,
  output logic [2*S-1:0][24:0]  n_caddr,   // the 64-byte beat in the channel (address bits 30:6)
  output logic [2*S-1:0]        n_wvalid,  // write data: a beat per write command, in their order
  input  logic [2*S-1:0]        n_wready,
  output logic [2*S-1:0][511:0] n_wdata,
  output logic [2*S-1:0][63:0]  n_wmask,   // 1 = write the byte
  input  logic [2*S-1:0]        n_rvalid,  // read data in read-command order, no backpressure
  input  logic [2*S-1:0][511:0] n_rdata,
  input  logic [2*S-1:0][15:0]  n_wdone,   // write beats the controller has taken (mod 2^16)
  input  logic [2*S-1:0]        n_err      // a channel's controller broke its port contract (sticky)
);
  import otpu_pkg::*;
  localparam int XR = 2;              // register stages each way across an SLR (S > 1)

  // ---- calibration and uncorrectable-ECC flags from the memory controllers' clock domains
  // (registers there; levels, each bit on its own)
  (* ASYNC_REG = "TRUE" *) logic [2*S-1:0] cal_s1, cal_s2, ded_s1, ded_s2;
  // CALIB0 / CALIB1: every slice's first / second channel
  logic [1:0] cal_all;
  always_comb begin
    cal_all = 2'b11;
    for (int k = 0; k < S; k++) cal_all &= cal_s2[2*k +: 2];
  end
  always_ff @(posedge clk) begin
    cal_s1 <= calib;
    cal_s2 <= cal_s1;
    ded_s1 <= ded;
    ded_s2 <= ded_s1;
  end

  // ---- die temperature: two flip-flops per bit, then a code is taken only when two
  // consecutive samples agree (it changes slowly: a sample caught mid-change is skipped). Valid
  // once the XADC (the block design's, on the core clock) has reported a reading; it does not
  // wait for the memory (the MIG builds' XADC was channel 0's, valid once it was calibrated).
  (* ASYNC_REG = "TRUE" *) logic [11:0] tmp_s1, tmp_s2;
  logic [11:0] tmp_s3, temp_q;
  logic        temp_v;
  always_ff @(posedge clk) begin
    tmp_s1 <= temp;
    tmp_s2 <= tmp_s1;
    tmp_s3 <= tmp_s2;
    if (rst) begin
      temp_q <= '0;
      temp_v <= 1'b0;
    end else if (tmp_s2 == tmp_s3 && tmp_s3 != '0) begin
      temp_q <= tmp_s3;
      temp_v <= 1'b1;
    end
  end

  // ---- I2C pin levels: two flip-flops each (the host samples them at microsecond pace)
  (* ASYNC_REG = "TRUE" *) logic [4:0] i2c_s1, i2c_s2;
  always_ff @(posedge clk) begin
    i2c_s1 <= i2c_pin;
    i2c_s2 <= i2c_s1;
  end

  // ---- control (slice 0's signals; the others' in g_slice)
  logic run, ld_start, ld_busy, halted, error, wait_to, wr_idle, rd_idle, a_inval;
  logic [31:0] ld_addr, ld_n;
  logic [S-1:0][31:0] icount;        // each slice's
  // every slice's, as the control block sees it: [0] slice 0's, [s] g_slice[s]'s
  logic [S-1:0] sl_ld_busy, sl_halted, sl_error, sl_wait_to, sl_wr_idle, sl_rd_idle;
  logic [S-1:0][1:0] sl_ret;         // instructions retired this cycle
  // the slices' own (q_*) and what each slice is given (x_*): the same at S = 1, two register
  // stages apart at S > 1 (g_xs)
  logic [S-1:0] q_ld_busy, q_halted, q_error, q_wait_to, q_wr_idle, q_rd_idle;
  logic [S-1:0][1:0] q_ret;
  logic [S-1:0][31:0] q_icount;
  logic [S-1:0] x_rst, x_srst, x_ld_start;
  logic [S-1:0][31:0] x_ld_addr, x_ld_n;
  logic [31:0] x_arg [S][8];
  // the free-running counters' increments: every channel's beats, every slice's instructions
  localparam int CW = $clog2(S * 2 + 1), IW = $clog2(2 * S + 1);
  logic [CW-1:0] dram_rd, dram_wr;
  logic [IW-1:0] instr;
  logic [31:0] arg [8];               // the run's arguments (ARG0..7: R8..R15 at the start)
  logic a_req, a_we, a_rvalid, a_rdy, b_req, b_tag, b_we, b_par, b_rvalid, b_rtag, b_rdy;
  logic [31:0] a_addr, a_wdata, a_rdata, a_rdata2, b_addr;
  logic [3:0]  a_be, sw_be;
  logic        sw_req, sw_rdy;
  logic [31:0] sw_addr, sw_wdata;
  logic [D/4-1:0] b_wmask;
  logic [D*8-1:0] b_wdata, b_rdata;

  perf_t pf, pf_c;                   // slice 0's, and as the control block sees it
  logic        tr_en, tr_stop, tr_clear, tr_busy;
  logic [31:0] tr_addr, tr_count, tr_drop;
  logic [63:0] tr_rdata;

  // the registers reset with rst (PERST#), the master also on a hot reset: an access in flight
  // then is completed here and its response dropped (otpu_axil_iso)
  logic [11:0] c_awaddr, c_araddr;
  logic [31:0] c_wdata, c_rdata;
  logic [3:0]  c_wstrb;
  logic [1:0]  c_bresp, c_rresp;
  logic c_awvalid, c_awready, c_wvalid, c_wready, c_bvalid, c_bready;
  logic c_arvalid, c_arready, c_rvalid, c_rready;
  otpu_axil_iso #(.AW(12)) u_iso (
    .clk, .m_rst(ctl_mrst), .s_rst(rst),
    .m_awaddr(s_ctl_awaddr), .m_awvalid(s_ctl_awvalid), .m_awready(s_ctl_awready),
    .m_wdata(s_ctl_wdata), .m_wstrb(s_ctl_wstrb), .m_wvalid(s_ctl_wvalid),
    .m_wready(s_ctl_wready), .m_bresp(s_ctl_bresp), .m_bvalid(s_ctl_bvalid),
    .m_bready(s_ctl_bready), .m_araddr(s_ctl_araddr), .m_arvalid(s_ctl_arvalid),
    .m_arready(s_ctl_arready), .m_rdata(s_ctl_rdata), .m_rresp(s_ctl_rresp),
    .m_rvalid(s_ctl_rvalid), .m_rready(s_ctl_rready),
    .s_awaddr(c_awaddr), .s_awvalid(c_awvalid), .s_awready(c_awready), .s_wdata(c_wdata),
    .s_wstrb(c_wstrb), .s_wvalid(c_wvalid), .s_wready(c_wready), .s_bresp(c_bresp),
    .s_bvalid(c_bvalid), .s_bready(c_bready), .s_araddr(c_araddr), .s_arvalid(c_arvalid),
    .s_arready(c_arready), .s_rdata(c_rdata), .s_rresp(c_rresp), .s_rvalid(c_rvalid),
    .s_rready(c_rready));

  otpu_ctrl #(.D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .LANES(LANES), .CORE_KHZ(CORE_KHZ), .BUILD_ID(BUILD_ID),
              .DDR_MTS(DDR_MTS), .TRACE_DEPTH(TRACE_DEPTH), .PQ_WIN(PQ_WIN), .HAS_TEMP(1'b1),
              .HAS_I2C(HAS_I2C), .CHASH(CHASH), .DSTEP(DSTEP), .HOSTCAL(HOSTCAL),
              .LDP(S > 1 ? 2 + 2 * XR : 2),  // LOAD to a slice's busy and back
              // each slice on its two channels' 2 GiB each (otpu_native_dram): 4 GiB
              .S(S), .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .ACT_BLOCKS(ACT_BLOCKS),
              .DRAM_LOG2(32), .NCH(2), .CH_LOG2(31), .MXU_IMPL(MXU_IMPL), .TMEM_IMPL(0),
              .VPU_CL(VPU_CL), .ULANES(ULANES), .PLATFORM(PLATFORM),
              .ARCH_HASH(ARCH_HASH)) u_ctrl (
    .clk, .rst,
    .s_awaddr(c_awaddr), .s_awvalid(c_awvalid), .s_awready(c_awready),
    .s_wdata(c_wdata), .s_wstrb(c_wstrb), .s_wvalid(c_wvalid),
    .s_wready(c_wready), .s_bresp(c_bresp), .s_bvalid(c_bvalid),
    .s_bready(c_bready), .s_araddr(c_araddr), .s_arvalid(c_arvalid),
    .s_arready(c_arready), .s_rdata(c_rdata), .s_rresp(c_rresp),
    .s_rvalid(c_rvalid), .s_rready(c_rready),
    .run, .ld_start, .arg, .ld_addr, .ld_n, .ld_busy, .halted, .error, .wait_to, .icount, .wr_idle,
    .axi_err(|n_err),                  // the channels' bridges (otpu_mem_ch)
    .calib(cal_all), .ecc_ded(|ded_s2),
    .b_rd(b_req && b_rdy && !b_we), .b_wr(b_req && b_rdy && b_we),
    .a_rd(a_req && a_rdy && !a_we), .a_wr(sw_req && sw_rdy), .b_wait(b_req && !b_rdy),
    .temp_v, .temp(temp_q),
    .mxu_busy(pf_c.sq.busy[U_MXU]), .mxu_mac(pf_c.mac), .mxu_starve(pf_c.starve),
    .vpu_busy(pf_c.sq.busy[U_VPU]),
    .qnt_busy(pf_c.sq.busy[U_Q]), .dma_busy(pf_c.sq.busy[U_DMA]), .tmem_deny(pf_c.deny),
    .dram_rd, .dram_wr,
    .dram_wait((b_req && !b_rdy) || (a_req && !a_rdy) || (sw_req && !sw_rdy)),
    .instr,
    .tr_en, .tr_stop, .tr_clear, .tr_addr, .tr_count, .tr_drop, .tr_busy, .tr_rdata,
    .i2c_lo, .i2c_in(i2c_s2));

  // ---- hardware trace
  if (TRACE_DEPTH != 0) begin : g_trace
    otpu_trace #(.DEPTH(TRACE_DEPTH), .QD(TRACE_QD), .WIN(WIN)) u_trace (
      .clk, .rst, .pf(pf_c), .en(tr_en && run), .stop(tr_stop), .clear(tr_clear), .raddr(tr_addr),
      .rdata(tr_rdata), .count(tr_count), .drop(tr_drop), .busy(tr_busy));
  end else begin : g_no_trace
    assign tr_rdata = '0;
    assign tr_count = '0;
    assign tr_drop = '0;
    assign tr_busy = 1'b0;
  end

  // ---- the slices (held in reset while RUN is 0) and the collective unit.
  // core_rst reaches ~15k flip-flops across the die: synthesis replicates it (a single copy's
  // net took 9.6 ns, the worst core_clk path at 100 MHz). A run starts from a quiet memory
  // adapter (go_ok: no read in flight, every write taken, sampled while RUN is 0 and until the
  // run starts): the adapter does not reset with RUN, so a run the host stopped leaves its reads
  // and writes going, and the next run's units would take the old reads' data
  (* max_fanout = 256 *) logic core_rst;
  logic go_ok;
  always_ff @(posedge clk) begin
    go_ok <= !rst && ((run && go_ok) || (rd_idle && wr_idle));
    core_rst <= rst || !run || !go_ok;
  end

  // the run's status from every slice
  assign ld_busy = |sl_ld_busy;
  assign halted  = &sl_halted;
  assign error   = |sl_error;
  assign wait_to = |sl_wait_to;
  assign wr_idle = &sl_wr_idle;
  assign rd_idle = &sl_rd_idle;
  assign q_ret[0] = pf.sq.ret;

  // the collective unit's ports (coll_*) and each slice's (sc_*): wired together at S = 1, the
  // hub and a port per slice between them at S > 1
  logic [S-1:0] coll_req;
  logic         coll_ack, coll_gnt;
  cmd_t         coll_cmds [S];
  logic [S-1:0][LANES-1:0]       coll_ren;
  logic [S-1:0][LANES-1:0][31:0] coll_raddr, coll_rdata;
  logic [LANES-1:0]              coll_wen;
  logic [LANES-1:0][31:0]        coll_waddr, coll_wdata;
  logic [S-1:0]                  sc_req, sc_ack, sc_gl, sc_gnt;
  cmd_t                          sc_cmd [S];
  logic [S-1:0][LANES-1:0]       sc_ren, sc_wen;
  logic [S-1:0][LANES-1:0][31:0] sc_raddr, sc_rdata, sc_waddr, sc_wdata;

  otpu_slice #(.SID(0), .S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS), .ACT_ROWS(ACT_ROWS),
               .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .FIFO_DEPTH(FIFO_DEPTH),
               .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL),
               .MXU_CL(MXU_CL), .VPU_CL(VPU_CL), .ULANES(ULANES), .PQ_WIN(PQ_WIN),
               .HAS_DSTEP(DSTEP)) u_slice (
    .clk, .sys_rst(x_srst[0]), .rst(x_rst[0]), .rinit(x_arg[0]), .ld_start(x_ld_start[0]),
    .ld_addr(x_ld_addr[0]), .ld_n(x_ld_n[0]),
    .ld_busy(q_ld_busy[0]), .a_rdy, .b_rdy, .sw_rdy, .wr_idle(q_wr_idle[0]),
    .rd_idle(q_rd_idle[0]),
    .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata, .a_rdata2,
    .sw_req, .sw_addr, .sw_wdata, .sw_be,
    .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag, .b_rdata,
    .coll_req(sc_req[0]), .coll_cmd(sc_cmd[0]), .coll_ack(sc_ack[0]),
    .coll_ren(sc_ren[0]), .coll_raddr(sc_raddr[0]), .coll_rdata(sc_rdata[0]),
    .coll_wen(sc_wen[0]), .coll_waddr(sc_waddr[0]), .coll_wdata(sc_wdata[0]),
    .coll_gnt_local(sc_gl[0]), .coll_gnt(sc_gnt[0]),
    .halted(q_halted[0]), .error(q_error[0]), .wait_to(q_wait_to[0]), .a_inval,
    .icount(q_icount[0]), .pf, .dump(1'b0));

  // the collective's reset through a register of its own, next to it: it leaves reset a cycle
  // after the slice, idle either way (133.33 MHz, 110ec6d: core_rst -> u_coll's state, 0 levels,
  // 97% route, +0.162 ns); a request can only come many cycles after reset (checked)
  logic coll_rst;
  always_ff @(posedge clk) coll_rst <= core_rst;
`ifndef SYNTHESIS
  always @(posedge clk)
    if (coll_rst && |coll_req) $fatal(1, "otpu_board: a collective request in reset");
`endif
  otpu_coll #(.S(S), .LANES(LANES)) u_coll (
    .clk, .rst(coll_rst), .req(coll_req), .cmds(coll_cmds), .gnt(coll_gnt), .ack(coll_ack),
    .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
    .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata));

  // ---- memory: the native adapter
  // port A's held beats go where the host may have written: between runs, a load, a WAITW
  otpu_native_dram #(.D(D), .CHASH(CHASH)) u_mem (
    .clk, .rst(x_srst[0]), .a_flush(x_rst[0] || x_ld_start[0] || a_inval),
    .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr), .a_wdata_x(a_wdata),
        .a_be_x(a_be), .a_rvalid, .a_rdata, .a_rdata2,
    .sw_rdy, .sw_req, .sw_addr, .sw_wdata, .sw_be,
    .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag,
    .b_rdata, .wr_idle(q_wr_idle[0]), .rd_idle(q_rd_idle[0]),
    .n_cvalid(n_cvalid[1:0]), .n_cready(n_cready[1:0]), .n_cwe(n_cwe[1:0]),
    .n_caddr(n_caddr[1:0]), .n_wvalid(n_wvalid[1:0]), .n_wready(n_wready[1:0]),
    .n_wdata(n_wdata[1:0]), .n_wmask(n_wmask[1:0]), .n_rvalid(n_rvalid[1:0]),
    .n_rdata(n_rdata[1:0]), .n_wdone(n_wdone[1:0]));

  // ---- slices 1 .. S-1, each on its two channels
  for (genvar s = 1; s < S; s++) begin : g_slice
    logic           a_req, a_we, a_rvalid, a_rdy, b_req, b_tag, b_we, b_par, b_rvalid, b_rtag;
    logic           b_rdy, a_inval, sw_req, sw_rdy;
    logic [31:0]    a_addr, a_wdata, a_rdata, a_rdata2, b_addr, sw_addr, sw_wdata;
    logic [3:0]     a_be, sw_be;
    logic [D/4-1:0] b_wmask;
    logic [D*8-1:0] b_wdata, b_rdata;
    perf_t          pf;
    assign q_ret[s] = pf.sq.ret;

    otpu_slice #(.SID(s), .S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS),
                 .ACT_ROWS(ACT_ROWS), .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS),
                 .FIFO_DEPTH(FIFO_DEPTH), .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB),
                 .MXU_IMPL(MXU_IMPL), .MXU_CL(MXU_CL), .VPU_CL(VPU_CL), .ULANES(ULANES),
                 .PQ_WIN(PQ_WIN), .HAS_DSTEP(DSTEP)) u_slice (
      .clk, .sys_rst(x_srst[s]), .rst(x_rst[s]), .rinit(x_arg[s]), .ld_start(x_ld_start[s]),
      .ld_addr(x_ld_addr[s]), .ld_n(x_ld_n[s]),
      .ld_busy(q_ld_busy[s]), .a_rdy, .b_rdy, .sw_rdy, .wr_idle(q_wr_idle[s]),
      .rd_idle(q_rd_idle[s]),
      .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata, .a_rdata2,
      .sw_req, .sw_addr, .sw_wdata, .sw_be,
      .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag, .b_rdata,
      .coll_req(sc_req[s]), .coll_cmd(sc_cmd[s]), .coll_ack(sc_ack[s]),
      .coll_ren(sc_ren[s]), .coll_raddr(sc_raddr[s]), .coll_rdata(sc_rdata[s]),
      .coll_wen(sc_wen[s]), .coll_waddr(sc_waddr[s]), .coll_wdata(sc_wdata[s]),
      .coll_gnt_local(sc_gl[s]), .coll_gnt(sc_gnt[s]),
      .halted(q_halted[s]), .error(q_error[s]), .wait_to(q_wait_to[s]), .a_inval,
      .icount(q_icount[s]), .pf, .dump(1'b0));

    otpu_native_dram #(.D(D), .CHASH(CHASH)) u_mem (
      .clk, .rst(x_srst[s]), .a_flush(x_rst[s] || x_ld_start[s] || a_inval),
      .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr), .a_wdata_x(a_wdata),
          .a_be_x(a_be), .a_rvalid, .a_rdata, .a_rdata2,
      .sw_rdy, .sw_req, .sw_addr, .sw_wdata, .sw_be,
      .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag,
      .b_rdata, .wr_idle(q_wr_idle[s]), .rd_idle(q_rd_idle[s]),
      .n_cvalid(n_cvalid[2*s +: 2]), .n_cready(n_cready[2*s +: 2]), .n_cwe(n_cwe[2*s +: 2]),
      .n_caddr(n_caddr[2*s +: 2]), .n_wvalid(n_wvalid[2*s +: 2]),
      .n_wready(n_wready[2*s +: 2]), .n_wdata(n_wdata[2*s +: 2]), .n_wmask(n_wmask[2*s +: 2]),
      .n_rvalid(n_rvalid[2*s +: 2]), .n_rdata(n_rdata[2*s +: 2]), .n_wdone(n_wdone[2*s +: 2]));
  end

  if (S == 1) begin : g_x1
    // ---- one slice: wired
    assign x_rst[0] = core_rst;
    assign x_srst[0] = rst;
    assign x_ld_start[0] = ld_start;
    assign x_ld_addr[0] = ld_addr;
    assign x_ld_n[0] = ld_n;
    assign x_arg[0] = arg;
    assign sl_ld_busy = q_ld_busy;
    assign sl_halted = q_halted;
    assign sl_error = q_error;
    assign sl_wait_to = q_wait_to;
    assign sl_wr_idle = q_wr_idle;
    assign sl_rd_idle = q_rd_idle;
    assign sl_ret = q_ret;
    assign icount = q_icount;
    assign pf_c = pf;
    always_comb begin
      dram_rd = '0;
      dram_wr = '0;
      instr = '0;
      for (int c = 0; c < 2 * S; c++) begin
        dram_rd += CW'(n_rvalid[c]);
        dram_wr += CW'(n_wvalid[c] && n_wready[c]);
      end
      for (int k = 0; k < S; k++) instr += IW'(sl_ret[k]);
    end
    assign coll_req = sc_req;
    assign coll_cmds = sc_cmd;
    assign sc_ack = coll_ack;
    assign sc_ren = coll_ren;
    assign sc_raddr = coll_raddr;
    assign coll_rdata = sc_rdata;
    assign sc_wen[0] = coll_wen;
    assign sc_waddr[0] = coll_waddr;
    assign sc_wdata[0] = coll_wdata;
    assign coll_gnt = &sc_gl;           // every slice's TMEM grant
    assign sc_gnt = coll_gnt;
  end else begin : g_xs
    // ---- S slices, each in its own SLR: two register stages each way between a slice and the
    // control block / the collective unit (otpu_xr: u_*_tx on the sending side, u_*_rx on the
    // receiving one; bcu1525_floorplan.tcl places them), so every signal crosses from a flip-flop
    // into one. Every slice's control goes through them, slice 0's too (the slices start
    // together); what comes back is XR cycles late, which the control block's LOAD window
    // (LDP) covers. The collective: otpu_coll_hub next to the unit, an otpu_coll_port per slice.
    localparam int L = LANES, CB = $bits(cmd_t), PB = $bits(perf_t);
    // down, per slice: pulses (ack, read, write; cleared by reset), the collective's read and
    // write, the control
    localparam int DW = 3 + L + 32 * L + L + 64 * L + 3 + 64 + 256;
    // up, per slice: levels / pulses (request, read data, pop; cleared by reset), the data, the
    // command, the status, the channels' beats this cycle
    localparam int UW = 3 + 32 * L + CB + 6 + 2 + 32 + 4;
    logic                   h_ack, h_wv;
    logic [S-1:0]           h_rv, h_req, h_dv, h_pop;
    logic [L-1:0]           h_rm, h_wm;
    logic [L-1:0][31:0]     h_ra, h_wa, h_wd;
    logic [S-1:0][L-1:0][31:0] h_dd;
    cmd_t                   h_cmd [S];
    logic [S-1:0][1:0]      sl_drd, sl_dwr;

    otpu_coll_hub #(.S(S), .LANES(L)) u_chub (
      .clk, .rst(coll_rst),
      .u_req(coll_req), .u_cmds(coll_cmds), .u_gnt(coll_gnt), .u_ack(coll_ack),
      .u_ren(coll_ren), .u_raddr(coll_raddr), .u_rdata(coll_rdata),
      .u_wen(coll_wen), .u_waddr(coll_waddr), .u_wdata(coll_wdata),
      .x_req(h_req), .x_cmds(h_cmd), .x_ack(h_ack), .x_rv(h_rv), .x_rm(h_rm), .x_ra(h_ra),
      .x_dv(h_dv), .x_dd(h_dd), .x_wv(h_wv), .x_wm(h_wm), .x_wa(h_wa), .x_wd(h_wd),
      .x_pop(h_pop));

    for (genvar s = 0; s < S; s++) begin : g_x
      logic [DW-1:0] dn_d, dn_m, dn_q;
      logic [UW-1:0] up_d, up_m, up_q;
      logic [255:0]  arg_d, arg_q;
      logic          p_dv, p_pop;
      logic [L-1:0][31:0] p_dd;
      logic [L-1:0]       p_rm, p_wm;
      logic [L-1:0][31:0] p_ra, p_wa, p_wd;
      logic          p_rv, p_wv, rx_crst;
      logic [1:0]    drd, dwr;
      for (genvar k = 0; k < 8; k++) begin : g_arg
        assign arg_d[32*k +: 32] = arg[k];
        assign x_arg[s][k] = arg_q[32*k +: 32];
      end
      assign dn_d = {arg_d, ld_n, ld_addr, ld_start, core_rst, rst, h_wd, h_wa, h_wm, h_ra, h_rm,
                     h_wv, h_rv[s], h_ack};
      otpu_xr #(.W(DW), .RW(3)) u_dn_tx (.clk, .rst(coll_rst), .d(dn_d), .q(dn_m));
      otpu_xr #(.W(DW), .RW(3)) u_dn_rx (.clk, .rst(rx_crst), .d(dn_m), .q(dn_q));
      assign {arg_q, x_ld_n[s], x_ld_addr[s], x_ld_start[s], rx_crst, x_srst[s], p_wd, p_wa,
              p_wm, p_ra, p_rm, p_wv, p_rv, sc_ack[s]} = dn_q;
      assign x_rst[s] = rx_crst;

      otpu_coll_port #(.LANES(L)) u_cport (
        .clk, .rst(x_rst[s]), .rv(p_rv), .rm(p_rm), .ra(p_ra), .wv(p_wv), .wm(p_wm), .wa(p_wa),
        .wd(p_wd), .dv(p_dv), .dd(p_dd), .pop(p_pop),
        .coll_ren(sc_ren[s]), .coll_raddr(sc_raddr[s]), .coll_rdata(sc_rdata[s]),
        .coll_wen(sc_wen[s]), .coll_waddr(sc_waddr[s]), .coll_wdata(sc_wdata[s]),
        .gnt_local(sc_gl[s]), .gnt(sc_gnt[s]));

      // the slice's channels' beats this cycle (DRAM_RD / DRAM_WR), counted on its side
      always_comb begin
        drd = '0;
        dwr = '0;
        for (int c = 2 * s; c < 2 * s + 2; c++) begin
          drd += 2'(n_rvalid[c]);
          dwr += 2'(n_wvalid[c] && n_wready[c]);
        end
      end
      assign up_d = {drd, dwr, q_icount[s], q_ret[s], q_ld_busy[s], q_halted[s], q_error[s],
                     q_wait_to[s], q_wr_idle[s], q_rd_idle[s], sc_cmd[s], p_dd, p_pop, p_dv,
                     sc_req[s]};
      otpu_xr #(.W(UW), .RW(3)) u_up_tx (.clk, .rst(x_rst[s]), .d(up_d), .q(up_m));
      otpu_xr #(.W(UW), .RW(3)) u_up_rx (.clk, .rst(coll_rst), .d(up_m), .q(up_q));
      assign {sl_drd[s], sl_dwr[s], icount[s], sl_ret[s], sl_ld_busy[s], sl_halted[s],
              sl_error[s], sl_wait_to[s], sl_wr_idle[s], sl_rd_idle[s], h_cmd[s], h_dd[s],
              h_pop[s], h_dv[s], h_req[s]} = up_q;

      if (s == 0) begin : g_pf          // slice 0's activity (the counters, the trace)
        logic [PB-1:0] pf_m;
        otpu_xr #(.W(PB)) u_pf_tx (.clk, .rst(1'b0), .d(pf), .q(pf_m));
        otpu_xr #(.W(PB)) u_pf_rx (.clk, .rst(1'b0), .d(pf_m), .q(pf_c));
      end
    end

    always_comb begin
      dram_rd = '0;
      dram_wr = '0;
      instr = '0;
      for (int k = 0; k < S; k++) begin
        dram_rd += CW'(sl_drd[k]);
        dram_wr += CW'(sl_dwr[k]);
        instr += IW'(sl_ret[k]);
      end
    end
  end

  // ---- LEDs: heartbeat, running, halted/error
  logic [26:0] hb;
  always_ff @(posedge clk) hb <= hb + 1;
  assign led = {halted && !error, run && !halted, hb[26]};
endmodule

// One register stage of an SLR crossing (otpu_board at S > 1: a pair per direction and slice, one
// stage on each side). The low RW bits (valid pulses, request levels) clear with rst; the others
// are data. max_fanout: a stage's reset bit drives a slice's whole core reset (as core_rst).
module otpu_xr #(
  parameter int W  = 1,
  parameter int RW = 0
) (
  input  logic         clk,
  input  logic         rst,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  (* max_fanout = 256 *) logic [W-1:0] r;     // r_reg: the floorplan's USER_SLL_REG cells
  assign q = r;
  always_ff @(posedge clk) begin
    r <= d;
    if (rst) for (int i = 0; i < RW; i++) r[i] <= 1'b0;
  end
endmodule
