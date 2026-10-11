// The systolic MXU's arithmetic cells (otpu_mxu IMPL 2) on Xilinx UltraScale / UltraScale+
// DSP48E2s (the BCU1525's VU9P). The same pipeline as the 7-series cells (rtl/tech/xilinx7, whose
// comments say why each register is where it is), on the DSP48E2's ports: the 27-bit D port and
// pre-adder, the multiplier's A input from the pre-adder (AMULTSEL "AD", PREADDINSEL "A": the
// DSP48E1's USE_DPORT), the 9-bit OPMODE (W = 0). INMODE has the DSP48E1's meaning (bit1 gates A,
// bit2 enables D, bit3 subtracts). rtl/tech/generic holds the equivalent behavioural models; the
// file list picks one (opentpu/target.py, docs/targets.md).

// One position of the systolic MXU (IMPL 2): a DSP48E2 with the activation in its B register
// (BREG 1), the two weight streams on A (low) and D (high), one selected by the pre-adder's input
// gates (INMODE: sel 0 -> A, 1 -> D), the product in M and the chain's running sum in P:
// P = PCIN + M (FIRST: P = M). A chain's LAST position leaves P unregistered: the column end's A
// / D input registers take its place (otpu_colend), so the route to the column end is not in
// front of its pre-adder and multiplier. The select is registered in the DSP (INMODEREG, loaded
// with en): sel is its next value, so the fabric's route to the DSP column ends at the INMODE
// register instead of before the pre-adder and multiplier (133.33 MHz, 812bb01: hi0 -> INMODE,
// 0 levels, 4.4 ns of route and a 2.4 ns setup, +0.174 ns). WREG registers the weights too (A
// and D registers, loaded with en: wlo and whi are their next values). rtl/tech/generic holds the
// equivalent behavioural model.
module otpu_pe #(parameter bit FIRST = 1'b0, parameter bit LAST = 1'b0, parameter bit WREG = 1'b0) (
  input  logic               clk,
  input  logic               en,
  input  logic [7:0]         act,
  input  logic [7:0]         wlo,               // WREG: the weights from the next cycle on
  input  logic [7:0]         whi,
  input  logic               sel,               // the select from the next cycle on
  input  logic [47:0]        pcin,
  output logic [47:0]        pcout,
  output logic signed [23:0] p
);
  logic [47:0] pf;
  DSP48E2 #(
    .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .AMULTSEL("AD"), .BMULTSEL("B"),
      .PREADDINSEL("A"), .USE_MULT("MULTIPLY"), .USE_WIDEXOR("FALSE"),
    .USE_SIMD("ONE48"), .AREG(WREG ? 1 : 0), .ACASCREG(WREG ? 1 : 0), .BREG(1), .BCASCREG(1),
    .CREG(0), .DREG(WREG ? 1 : 0),
    .ADREG(0), .MREG(1), .PREG(LAST ? 0 : 1), .INMODEREG(1), .OPMODEREG(0), .ALUMODEREG(0),
    .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
    .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
  ) u_dsp (
    .CLK(clk),
    .A({{22{wlo[7]}}, wlo}), .B({{10{act[7]}}, act}), .C(48'd0), .D({{19{whi[7]}}, whi}),
    .INMODE({2'b00, sel, sel, 1'b0}), .OPMODE(FIRST ? 9'b00_000_01_01 : 9'b00_001_01_01),
    .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
    .CEA1(1'b0), .CEA2(WREG ? en : 1'b0), .CEB1(en), .CEB2(en), .CEC(1'b0),
    .CED(WREG ? en : 1'b0), .CEAD(1'b0),
    .CEM(en), .CEP(LAST ? 1'b0 : en), .CEALUMODE(1'b0), .CECTRL(1'b0), .CECARRYIN(1'b0),
    .CEINMODE(en),
    .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
    .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
    .ACIN(30'd0), .BCIN(18'd0), .PCIN(pcin), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
    .ACOUT(), .BCOUT(), .PCOUT(pcout), .P(pf), .CARRYCASCOUT(), .MULTSIGNOUT(), .CARRYOUT(),
    .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT());
  assign p = pf[23:0];
endmodule


// The systolic MXU's column end (IMPL 2): four DSP48E2s, one per sub-block b. DSP b registers the
// sub-block's chain ends (a[b], d[b]: the chains' last P, which otpu_pe LAST leaves unregistered)
// in its A and D registers, sums them in its pre-adder, multiplies by m_b (its B register:
// mb[4b+3:4b], unsigned, loaded with a / d) into M (v[b]); the block sum runs down the cascade in
// the same cycle, P = PCIN + M (DSP 0: P = M), and DSP 3 registers it (PREG):
//   s = v0 + v1 + v2 + v3   (one cycle after v, as a fabric or inferred sum would)
// The cascade keeps the sum off the fabric: Vivado mapped the inferred sum into three DSP ALUs
// joined through the C port (two fabric routes, 6.76 ns: -0.479 ns at 133.33 MHz). The input
// registers keep the route from the chain ends (3.5 ns in the OOC) out of the multiplier's cycle:
// the chain ends' P -> v was 0.008 ns from failing out of context at 133.33 MHz. Exact: every
// value fits 48 bits, s is the sum's low SW bits. rtl/tech/generic holds the
// equivalent behavioural model.
module otpu_colend #(parameter int SW = 23) (
  input  logic                 clk,
  input  logic                 en,
  input  logic [3:0][23:0]     a,        // sub-block b's chain ends: the pre-adder's A ...
  input  logic [3:0][23:0]     d,        // ... and D (signed)
  input  logic [15:0]          mb,       // m_3 .. m_0, with a / d
  output logic signed [SW-1:0] s
);
  // the DSP48E1's pre-adder width (25 bits; the DSP48E2's is 27): every technology takes
  // the same parameters
  initial if (SW > 25) $fatal(1, "otpu_colend: SW %0d > 25", SW);
  logic [3:0][47:0] pc, pf;
  for (genvar b = 0; b < 4; b++) begin : g_d
    DSP48E2 #(
      .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .AMULTSEL("AD"), .BMULTSEL("B"),
      .PREADDINSEL("A"), .USE_MULT("MULTIPLY"), .USE_WIDEXOR("FALSE"),
      .USE_SIMD("ONE48"), .AREG(1), .ACASCREG(1), .BREG(1), .BCASCREG(1), .CREG(0), .DREG(1),
      .ADREG(0), .MREG(1), .PREG(b == 3 ? 1 : 0), .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
      .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
      .AUTORESET_PATDET("NO_RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h0),
      .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
    ) u_dsp (
      .CLK(clk),
      .A({{6{a[b][23]}}, a[b]}), .B({14'd0, mb[4*b +: 4]}), .C(48'd0), .D({{3{d[b][23]}}, d[b]}),
      .INMODE(5'b00100), .OPMODE(b == 0 ? 9'b00_000_01_01 : 9'b00_001_01_01),
      .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
      .CEA1(en), .CEA2(en), .CEB1(en), .CEB2(en), .CEC(1'b0), .CED(en), .CEAD(1'b0),
      .CEM(en), .CEP(b == 3 ? en : 1'b0), .CEALUMODE(1'b0), .CECTRL(1'b0), .CECARRYIN(1'b0),
      .CEINMODE(1'b0),
      .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
      .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
      .ACIN(30'd0), .BCIN(18'd0), .PCIN(b == 0 ? 48'd0 : pc[b == 0 ? 0 : b - 1]),
      .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
      .ACOUT(), .BCOUT(), .PCOUT(pc[b]), .P(pf[b]), .CARRYCASCOUT(), .MULTSIGNOUT(),
      .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT());
  end
  assign s = $signed(pf[3][SW-1:0]);
endmodule
