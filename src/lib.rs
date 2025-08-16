//! RISC-V 32-Bit encoding/decoding crate

#![no_std]
#![warn(
    anonymous_parameters,
    missing_copy_implementations,
    missing_debug_implementations,
    nonstandard_style,
    rust_2018_idioms,
    single_use_lifetimes,
    trivial_casts,
    trivial_numeric_casts,
    unreachable_pub,
    unused_extern_crates,
    unused_qualifications,
    variant_size_differences
)]

use I32::*;
use Reg::*;

/// A RISC-V Register
#[repr(u8)]
#[derive(Eq, Ord, Hash, Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum Reg {
    /// SPECIAL: Always 0
    ZERO = 0u8,
    /// Return address
    RA = 1,
    /// SPECIAL: Stack pointer
    SP = 2,
    /// Global pointer
    GP = 3,
    /// Thread pointer
    TP = 4,
    /// Temporary
    T0 = 5,
    /// Temporary
    T1 = 6,
    /// Temporary
    T2 = 7,
    /// Saved (Frame Pointer)
    S0 = 8,
    /// Saved
    S1 = 9,
    /// Function arguments / Return values
    A0 = 10,
    /// Function arguments / Return values
    A1 = 11,
    /// Function arguments
    A2 = 12,
    /// Function arguments
    A3 = 13,
    /// Function arguments
    A4 = 14,
    /// Function arguments
    A5 = 15,
    /// Function arguments
    A6 = 16,
    /// Function arguments
    A7 = 17,
    /// Saved
    S2 = 18,
    /// Saved
    S3 = 19,
    /// Saved
    S4 = 20,
    /// Saved
    S5 = 21,
    /// Saved
    S6 = 22,
    /// Saved
    S7 = 23,
    /// Saved
    S8 = 24,
    /// Saved
    S9 = 25,
    /// Saved
    S10 = 26,
    /// Saved
    S11 = 27,
    /// Temporary
    T3 = 28,
    /// Temporary
    T4 = 29,
    /// Temporary
    T5 = 30,
    /// Temporary
    T6 = 31,
}

impl Reg {
    #[inline(always)]
    pub const fn from_u32(reg: u32) -> Self {
        debug_assert!(reg < 32);
        unsafe { core::mem::transmute(reg as u8) }
    }
}

impl From<u32> for Reg {
    #[inline(always)]
    fn from(reg: u32) -> Self {
        Self::from_u32(reg)
    }
}

/// Error types when converting `u32` to `I`
#[derive(Debug, Clone, Copy)]
pub enum ConversionError {
    /// Unknown funct3 field
    UnknownFunct3(u32),

    /// Unknown funct7 field
    UnknownFunct7(u32),

    /// Unknown funct3 or funct7 field
    UnknownFunct3Funct7(u32, u32),

    /// Unknown funct3 or funct5 field
    UnknownFunct3Funct5(u32, u32),

    /// Unknown Environment Control Transfer
    UnknownEnvCtrlTransfer,

    /// Unknown opcode
    UnknownOpcode(u32),
}

/// Memory ordering bits for A-extension instructions (aq: acquire, rl: release).
#[repr(u8)]
#[derive(Eq, Copy, Clone, Debug, PartialEq)]
pub enum AqRl {
    /// No ordering constraints       (aq=0, rl=0).
    None           = 0b00,
    /// Release semantics             (aq=0, rl=1).
    Release        = 0b01,
    /// Acquire semantics             (aq=1, rl=0).
    Acquire        = 0b10,
    /// Acquire and release semantics (aq=1, rl=1).
    AcquireRelease = 0b11,
}

impl AqRl {
    /// Returns the 2-bit aq/rl value as a `u32` for instruction encoding.
    #[inline(always)]
    pub const fn as_u32(self) -> u32 {
        self as u8 as u32
    }

    #[inline(always)]
    pub const fn from_u32(with: u32) -> Self {
        debug_assert!(with < 4);
        unsafe { core::mem::transmute(with as u8) }
    }
}

/// An assembly instruction (im is limited to 12 bits)
///
/// This enum does not enforce a specific shift amount (shamt) range.
/// This means a single enum variant can encode both RV32 and RV64 forms.
/// For RV32: valid shamt is 0–31 (immediate) or lower 5 bits of rs2 (register shift).
/// For RV64: valid shamt is 0–63 (immediate) or lower 6 bits of rs2 (register shift).
#[allow(clippy::enum_variant_names)]
#[allow(non_camel_case_types)]
#[allow(missing_docs)]
#[derive(Copy, Clone, Debug)]
pub enum I32 {
    //// One of 40 User mode instructions in the RV32I Base Instruction Set ////
    /// U: Set upper 20 bits to immediate value
    LUI { d: Reg, im: i32 },
    /// U: Add upper 20 bits to immediate value in program counter
    AUIPC { d: Reg, im: i32 },
    /// UJ: Jump and Link Relative
    JAL { d: Reg, im: i32 },
    /// I: Jump and Link, Register
    JALR { d: Reg, s: Reg, im: i16 },
    /// SB: 12-bit immediate offset Branch on Equal
    BEQ { s1: Reg, s2: Reg, im: i16 },
    /// SB: 12-bit immediate offset Branch on Not Equal
    BNE { s1: Reg, s2: Reg, im: i16 },
    /// SB: 12-bit immediate offset Branch on Less Than
    BLT { s1: Reg, s2: Reg, im: i16 },
    /// SB: 12-bit immediate offset Branch on Greater Than Or Equal To
    BGE { s1: Reg, s2: Reg, im: i16 },
    /// SB: 12-bit immediate offset Branch on Less Than (Unsigned)
    BLTU { s1: Reg, s2: Reg, im: i16 },
    /// SB: 12-bit immediate offset Branch on Greater Than Or Equal To (Unsigned)
    BGEU { s1: Reg, s2: Reg, im: i16 },
    /// I: Load Byte (`R[d]: M[R[s] + im]`)
    LB { d: Reg, s: Reg, im: i16 },
    /// I: Load Half-Word (`R[d]: M[R[s] + im]`)
    LH { d: Reg, s: Reg, im: i16 },
    /// I: Load Word (`R[d]: M[R[s] + im]`)
    LW { d: Reg, s: Reg, im: i16 },
    /// I: Load Byte Unsigned (`R[d]: M[R[s] + im]`)
    LBU { d: Reg, s: Reg, im: i16 },
    /// I: Load Half Unsigned (`R[d]: M[R[s] + im]`)
    LHU { d: Reg, s: Reg, im: i16 },
    /// S: Store Byte
    SB { s1: Reg, s2: Reg, im: i16 },
    /// S: Store Half Word
    SH { s1: Reg, s2: Reg, im: i16 },
    /// S: Store Word
    SW { s1: Reg, s2: Reg, im: i16 },
    /// I: Add Immediate (`R[d]: R[s] + im`)
    ADDI { d: Reg, s: Reg, im: i16 },
    /// I: Set 1 on Less Than, 0 Otherwise Immediate
    SLTI { d: Reg, s: Reg, im: i16 },
    /// I: Set 1 on Less Than, 0 Otherwise Immediate Unsigned
    SLTIU { d: Reg, s: Reg, im: i16 },
    /// I: Xor Immediate
    XORI { d: Reg, s: Reg, im: i16 },
    /// I: Or Immediate
    ORI { d: Reg, s: Reg, im: i16 },
    /// I: And Immediate
    ANDI { d: Reg, s: Reg, im: i16 },
    /// I: Logical Left Shift Immediate
    SLLI { d: Reg, s: Reg, shamt: u8 },
    /// I: Logical Right Shift Immediate
    SRLI { d: Reg, s: Reg, shamt: u8 },
    /// I: Arithmetic Shift Right Immediate (See SRA).
    SRAI { d: Reg, s: Reg, shamt: u8 },
    /// R: Logical Left Shift
    SLL { d: Reg, s1: Reg, s2: Reg },
    /// R: Logical Right Shift
    SRL { d: Reg, s1: Reg, s2: Reg },
    /// R: Arithmetic Shift Right (Sign Bit Copied Rather Than Filling In Zeros)
    SRA { d: Reg, s1: Reg, s2: Reg },
    /// R: Add (`R[d]: R[s1] + R[s2]`)
    ADD { d: Reg, s1: Reg, s2: Reg },
    /// R: Subtract (`R[d]: R[s1] - R[s2]`)
    SUB { d: Reg, s1: Reg, s2: Reg },
    /// R: Set 1 on Less Than, 0 Otherwise
    SLT { d: Reg, s1: Reg, s2: Reg },
    /// R: Set 1 on Less Than, 0 Otherwise Unsigned
    SLTU { d: Reg, s1: Reg, s2: Reg },
    /// R: Xor
    XOR { d: Reg, s1: Reg, s2: Reg },
    /// R: Or
    OR { d: Reg, s1: Reg, s2: Reg },
    /// R: And
    AND { d: Reg, s1: Reg, s2: Reg },
    /// I: Invoke a system call (Registers defined by ABI, not hardware)
    ECALL {},
    /// I: Debugger Breakpoint
    EBREAK {},
    /// I: Fence (Immediate Is Made Up Of Ordered High Order To Low Order Bits:)
    /// - fm(4), PI(1), PO(1), PR(1), PW(1), SI(1), SO(1), SR(1), SW(1)
    FENCE { im: i16 },

    //// M-extension instructions (Version 2.1) ////
    /// R: Multiply (`R[d]: (R[s1] * R[s2])[31:0]`) - Returns lower 32 bits of product
    MUL { d: Reg, s1: Reg, s2: Reg },
    /// R: Multiply High Signed x Signed (`R[d]: (R[s1] * R[s2])[63:32]`) - Returns upper 32 bits of signed x signed product
    MULH { d: Reg, s1: Reg, s2: Reg },
    /// R: Multiply High Signed x Unsigned (`R[d]: (R[s1] * R[s2])[63:32]`) - Returns upper 32 bits of signed x unsigned product
    MULHSU { d: Reg, s1: Reg, s2: Reg },
    /// R: Multiply High Unsigned x Unsigned (`R[d]: (R[s1] * R[s2])[63:32]`) - Returns upper 32 bits of unsigned x unsigned product
    MULHU { d: Reg, s1: Reg, s2: Reg },
    /// R: Divide (`R[d]: R[s1] / R[s2]`) - Signed division
    DIV { d: Reg, s1: Reg, s2: Reg },
    /// R: Divide Unsigned (`R[d]: R[s1] / R[s2]`) - Unsigned division
    DIVU { d: Reg, s1: Reg, s2: Reg },
    /// R: Remainder (`R[d]: R[s1] % R[s2]`) - Signed remainder
    REM { d: Reg, s1: Reg, s2: Reg },
    /// R: Remainder Unsigned (`R[d]: R[s1] % R[s2]`) - Unsigned remainder
    REMU { d: Reg, s1: Reg, s2: Reg },

    //// A-extension Load-Reserved/Store-Conditional Instructions ////
    /// R: Load-Reserved Word (`R[d]: M[R[s1]]`) - Loads 32-bit word, reserves address for SC
    LR_W { d: Reg, s1: Reg, aqrl: AqRl },
    /// R: Store-Conditional Word (`R[d]: success/fail, M[R[s1]]: R[s2]`) - Conditionally stores if reservation valid
    SC_W { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },

    //// A-extension Atomic Memory Operations (32-bit) ////
    /// R: Atomic Add Word (`R[d]: M[R[s1]], M[R[s1]]: M[R[s1]] + R[s2]`) - Atomically adds s2 to memory, returns original value
    AMOADD_W { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic Swap Word (`R[d]: M[R[s1]], M[R[s1]]: R[s2]`) - Atomically swaps s2 with memory, returns original value
    AMOSWAP_W { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic AND Word (`R[d]: M[R[s1]], M[R[s1]]: M[R[s1]] & R[s2]`) - Atomically ANDs s2 with memory, returns original value
    AMOAND_W { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic OR Word (`R[d]: M[R[s1]], M[R[s1]]: M[R[s1]] | R[s2]`) - Atomically ORs s2 with memory, returns original value
    AMOOR_W { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic XOR Word (`R[d]: M[R[s1]], M[R[s1]]: M[R[s1]] ^ R[s2]`) - Atomically XORs s2 with memory, returns original value
    AMOXOR_W { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic Max Word (`R[d]: M[R[s1]], M[R[s1]]: max(M[R[s1]], R[s2])`) - Atomically stores signed max, returns original value
    AMOMAX_W { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic Min Word (`R[d]: M[R[s1]], M[R[s1]]: min(M[R[s1]], R[s2])`) - Atomically stores signed min, returns original value
    AMOMIN_W { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic Max Unsigned Word (`R[d]: M[R[s1]], M[R[s1]]: maxu(M[R[s1]], R[s2])`) - Atomically stores unsigned max, returns original value
    AMOMAXU_W { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic Min Unsigned Word (`R[d]: M[R[s1]], M[R[s1]]: minu(M[R[s1]], R[s2])`) - Atomically stores unsigned min, returns original value
    AMOMINU_W { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
}

impl PartialEq for I32 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.into_u32() == other.into_u32()
    }
}

impl I32 {
    /// - funct7: 7
    /// - src2:   5
    /// - src1:   5
    /// - funct3: 3
    /// - dst:    5
    /// - opcode: 7
    #[inline(always)]
    pub const fn r(
        opcode: u32,
        d: Reg,
        funct3: u32,
        s1: Reg,
        s2: Reg,
        funct7: u32,
    ) -> u32 {
        let dst  = d  as u32;
        let src1 = s1 as u32;
        let src2 = s2 as u32;
        let mut out = opcode;
        out |= dst << 7;
        out |= funct3 << 12;
        out |= src1 << 15;
        out |= src2 << 20;
        out |= funct7 << 25;
        out
    }

    #[inline(always)]
    pub const fn from_r(instruction: u32) -> (Reg, u32, Reg, Reg, u32) {
        let d = Reg::from_u32((instruction & (0b11111 << 7)) >> 7);
        let funct3 = (instruction & (0b111 << 12)) >> 12;
        let s1 = Reg::from_u32((instruction & (0b11111 << 15)) >> 15);
        let s2 = Reg::from_u32((instruction & (0b11111 << 20)) >> 20);
        let funct7 = instruction >> 25;
        (d, funct3, s1, s2, funct7)
    }

    /// - im:    12
    /// - src:    5
    /// - funct3: 3
    /// - dst:    5
    /// - opcode  7
    #[inline(always)]
    pub const fn i(opcode: u32, d: Reg, funct3: u32, s: Reg, im: i16) -> u32 {
        let im  = (im as u16) as u32;
        let dst = d as u32;
        let src = s as u32;
        let mut out = opcode;
        out |= dst << 7;
        out |= funct3 << 12;
        out |= src << 15;
        out |= im << 20;
        out
    }

    #[inline(always)]
    pub const fn from_i(instruction: u32) -> (Reg, u32, Reg, i16) {
        let d = Reg::from_u32((instruction & (0b11111 << 7)) >> 7);
        let funct3 = (instruction & (0b111 << 12)) >> 12;
        let s = Reg::from_u32((instruction & (0b11111 << 15)) >> 15);
        let im_bits = (instruction >> 20) & 0xFFF;
        let im = if (im_bits & 0x800) != 0 {
            // negative: sign extend
            (im_bits | 0xFFFFF000) as i32
        } else {
            im_bits as i32
        } as i16;
        (d, funct3, s, im)
    }

    /// - funct7: 7
    /// - im:    5
    /// - src:    5
    /// - funct3: 3
    /// - dst:    5
    /// - opcode  7
    #[inline(always)]
    pub const fn i7(
        opcode: u32,
        d: Reg,
        funct3: u32,
        s: Reg,
        im: u8,
        funct7: u32,
    ) -> u32 {
        let im = im as u32;
        let dst = d as u32;
        let src = s as u32;
        let mut out = opcode;
        out |= dst << 7;
        out |= funct3 << 12;
        out |= src << 15;
        out |= im << 20;
        out |= funct7 << 25;
        out
    }

    #[inline(always)]
    pub const fn from_i7(instruction: u32) -> (Reg, u32, Reg, u8, u32) {
        let d = Reg::from_u32((instruction & (0b11111 << 7)) >> 7);
        let funct3 = (instruction & (0b111 << 12)) >> 12;
        let s = Reg::from_u32((instruction & (0b11111 << 15)) >> 15);
        let im = ((instruction & (0b11111 << 20)) >> 20) as u8;
        let funct7 = instruction >> 25;
        (d, funct3, s, im, funct7)
    }

    /// - im_h:  7
    /// - src2:   5
    /// - src1:   5
    /// - funct3: 3
    /// - im_l:  5
    /// - opcode  7
    pub const fn s(opcode: u32, funct3: u32, s1: Reg, s2: Reg, im: i16) -> u32 {
        let im  = (im as u16) as u32;
        let src1 = s1 as u32;
        let src2 = s2 as u32;
        let mut out = opcode;
        out |= (im & 0b11111) << 7;
        out |= funct3 << 12;
        out |= src1 << 15;
        out |= src2 << 20;
        out |= (im >> 5) << 25;
        out
    }

    #[inline(always)]
    pub const fn from_s(instruction: u32) -> (u32, Reg, Reg, i16) {
        let mut im = ((instruction & (0b11111 << 7)) >> 7) as u16;
        let funct3 = (instruction & (0b111 << 12)) >> 12;
        let s1 = Reg::from_u32((instruction & (0b11111 << 15)) >> 15);
        let s2 = Reg::from_u32((instruction & (0b11111 << 20)) >> 20);
        im |= ((instruction >> 25) as u16) << 5;
        let im_signed = if (im & 0x800) != 0 {
            // negative: sign extend
            (im | 0xF000) as i16
        } else {
            im as i16
        };
        (funct3, s1, s2, im_signed)
    }

    #[inline]
    pub const fn amo(opcode: u32, rd: Reg, funct3: u32, rs1: Reg, rs2: Reg, aqrl: AqRl, funct5: u32) -> u32 {
        (funct5 << 27) |
        ((aqrl.as_u32()) << 25) |
        ((rs2 as u32) << 20) |
        ((rs1 as u32) << 15) |
        (funct3 << 12) |
        ((rd as u32) << 7) |
        opcode
    }

    #[inline]
    pub const fn from_amo(inst: u32) -> (Reg, u32, Reg, Reg, AqRl, u32) {
        let rd = Reg::from_u32((inst >> 7) & 0x1f);
        let funct3 = (inst >> 12) & 0x7;
        let rs1 = Reg::from_u32((inst >> 15) & 0x1f);
        let rs2 = Reg::from_u32((inst >> 20) & 0x1f);
        let aqrl = AqRl::from_u32((inst >> 25) & 0x3);
        let funct5 = (inst >> 27) & 0x1f;
        (rd, funct3, rs1, rs2, aqrl, funct5)
    }

    /// - im:    20 (simm20)
    /// - dst:    5
    /// - opcode: 7
    #[inline(always)]
    pub const fn u(opcode: u32, d: Reg, im: i32) -> u32 {
        // mask to lower 20 bits (two's complement if im negative)
        let im20 = (im as u32) & 0xFFFFF;
        let dst = d as u32;
        opcode | (dst << 7) | (im20 << 12)
    }

    #[inline(always)]
    pub const fn from_u(instruction: u32) -> (Reg, i32) {
        let d = Reg::from_u32((instruction >> 7) & 0b11111);
        // extract 20-bit field
        let im20 = ((instruction >> 12) & 0xFFFFF) as i32;
        // sign-extend 20 bits into i32
        let imm_signed = (im20 << 12) >> 12;
        (d, imm_signed)
    }

    /// - im_b (branch offset, bytes): 13-bit signed, LSB must be 0 (2-byte alignment)
    /// - Layout into instr bits:
    ///   imm[12]   -> bit 31
    ///   imm[10:5] -> bits 30:25
    ///   imm[4:1]  -> bits 11:8
    ///   imm[11]   -> bit 7
    /// - src2:   5
    /// - src1:   5
    /// - funct3: 3
    /// - opcode: 7
    #[inline(always)]
    pub const fn b(opcode: u32, funct3: u32, s1: Reg, s2: Reg, im_b: i16) -> u32 {
        let imm = (im_b as i32 as u32) & 0x1FFF; // keep 13 bits
        let src1 = s1 as u32;
        let src2 = s2 as u32;

        let mut out = opcode;
        out |= ((imm >> 11) & 0x1) << 7;    // imm[11]   -> bit 7
        out |= ((imm >> 1)  & 0xF) << 8;    // imm[4:1]  -> bits 11:8
        out |= funct3 << 12;
        out |= src1 << 15;
        out |= src2 << 20;
        out |= ((imm >> 5)  & 0x3F) << 25;  // imm[10:5] -> bits 30:25
        out |= ((imm >> 12) & 0x1) << 31;   // imm[12]   -> bit 31
        out
    }

    #[inline(always)]
    pub const fn from_b(instruction: u32) -> (u32, Reg, Reg, i16) {
        let mut imm = 0u32;

        // rebuild 13-bit immediate
        imm |= ((instruction >> 7)  & 0x1)  << 11; // bit 7 -> imm[11]
        imm |= ((instruction >> 8)  & 0xF)  << 1;  // bits 11:8 -> imm[4:1]
        imm |= ((instruction >> 25) & 0x3F) << 5;  // bits 30:25 -> imm[10:5]
        imm |= ((instruction >> 31) & 0x1)  << 12; // bit 31 -> imm[12]

        // Sign-extend 13-bit to i16
        let imm_signed = if (imm & (1 << 12)) != 0 {
            (imm | 0xFFFF_E000) as i32
        } else {
            imm as i32
        } as i16;

        let funct3 = (instruction & (0b111 << 12)) >> 12;
        let s1 = Reg::from_u32((instruction & (0b11111 << 15)) >> 15);
        let s2 = Reg::from_u32((instruction & (0b11111 << 20)) >> 20);
        (funct3, s1, s2, imm_signed)
    }

    /// - im_j (jump offset, bytes): 21-bit signed, LSB must be 0 (2-byte alignment)
    /// - Layout into instr bits (rd + opcode already standard):
    ///   imm[20]    -> bit 31
    ///   imm[10:1]  -> bits 30:21
    ///   imm[11]    -> bit 20
    ///   imm[19:12] -> bits 19:12
    /// - dst: 5
    /// - opcode: 7
    #[inline(always)]
    pub const fn j(opcode: u32, d: Reg, im_j: i32) -> u32 {
        // treat im_j as 21-bit signed (bits 20:0), bit 0 must be zero.
        let imm = (im_j as u32) & 0x001F_FFFF; // keep 21 bits
        let dst = d as u32;
        let mut out = opcode;
        out |= dst << 7;
        // imm fields
        out |= imm & 0x000F_F000;           // imm[19:12] -> bits 19:12 (already aligned)
        out |= ((imm >> 11) & 0x1)   << 20; // imm[11]    -> bit 20
        out |= ((imm >> 1)  & 0x3FF) << 21; // imm[10:1]  -> bits 30:21
        out |= ((imm >> 20) & 0x1)   << 31; // imm[20]    -> bit 31
        out
    }

    #[inline(always)]
    pub const fn from_j(instruction: u32) -> (Reg, i32) {
        let d = Reg::from_u32((instruction & (0b11111 << 7)) >> 7);
        let mut imm = 0u32;
        imm |= instruction & 0x000F_F000;          // bits 19:12 -> imm[19:12]
        imm |= ((instruction >> 20) & 0x1) << 11;  // bit 20     -> imm[11]
        imm |= ((instruction >> 21) & 0x3FF) << 1; // bits 30:21 -> imm[10:1]
        imm |= ((instruction >> 31) & 0x1) << 20;  // bit 31     -> imm[20]

        // sign-extend 21-bit to i32
        let imm_signed = if (imm & (1 << 20)) != 0 {
            imm | 0xFFE0_0000
        } else {
            imm
        };

        (d, imm_signed as _)
    }

    #[inline]
    pub const fn into_u32(self) -> u32 {
        match self {
            LUI { d, im }       => I32::u(0b0110111,  d,     im),
            AUIPC { d, im }     => I32::u(0b0010111,  d,     im),
            JAL { d, im }       => I32::u(0b1101111,  d,     im),
            JALR { d, s, im }   => I32::i(0b1100111,  d,     0b000, s,    im),
            BEQ { s1, s2, im }  => I32::s(0b1100011,  0b000, s1,    s2,   im),
            BNE { s1, s2, im }  => I32::s(0b1100011,  0b001, s1,    s2,   im),
            BLT { s1, s2, im }  => I32::s(0b1100011,  0b100, s1,    s2,   im),
            BGE { s1, s2, im }  => I32::s(0b1100011,  0b101, s1,    s2,   im),
            BLTU { s1, s2, im } => I32::s(0b1100011,  0b110, s1,    s2,   im),
            BGEU { s1, s2, im } => I32::s(0b1100011,  0b111, s1,    s2,   im),
            LB { d, s, im }     => I32::i(0b0000011,  d,     0b000, s,    im),
            LH { d, s, im }     => I32::i(0b0000011,  d,     0b001, s,    im),
            LW { d, s, im }     => I32::i(0b0000011,  d,     0b010, s,    im),
            LBU { d, s, im }    => I32::i(0b0000011,  d,     0b100, s,    im),
            LHU { d, s, im }    => I32::i(0b0000011,  d,     0b101, s,    im),
            SB { s1, s2, im }   => I32::s(0b0100011,  0b000, s1,    s2,   im),
            SH { s1, s2, im }   => I32::s(0b0100011,  0b001, s1,    s2,   im),
            SW { s1, s2, im }   => I32::s(0b0100011,  0b010, s1,    s2,   im),
            ADDI { d, s, im }   => I32::i(0b0010011,  d,     0b000, s,    im),
            SLTI { d, s, im }   => I32::i(0b0010011,  d,     0b010, s,    im),
            SLTIU { d, s, im }  => I32::i(0b0010011,  d,     0b011, s,    im),
            XORI { d, s, im }   => I32::i(0b0010011,  d,     0b100, s,    im),
            ORI { d, s, im }    => I32::i(0b0010011,  d,     0b110, s,    im),
            ANDI { d, s, im }   => I32::i(0b0010011,  d,     0b111, s,    im),
            SLLI { d, s, shamt: im } => I32::i7(0b0010011, d,     0b001, s,    im, 0b0000000),
            SRLI { d, s, shamt: im } => I32::i7(0b0010011, d,     0b101, s,    im, 0b0000000),
            SRAI { d, s, shamt: im } => I32::i7(0b0010011, d,     0b101, s,    im, 0b0100000),
            ADD { d, s1, s2 }   => I32::r(0b0110011, d, 0b000, s1, s2, 0b0000000),
            SUB { d, s1, s2 }   => I32::r(0b0110011, d, 0b000, s1, s2, 0b0100000),
            SLL { d, s1, s2 }   => I32::r(0b0110011, d, 0b001, s1, s2, 0b0000000),
            SLT { d, s1, s2 }   => I32::r(0b0110011, d, 0b010, s1, s2, 0b0000000),
            SLTU { d, s1, s2 }  => I32::r(0b0110011, d, 0b011, s1, s2, 0b0000000),
            XOR { d, s1, s2 }   => I32::r(0b0110011, d, 0b100, s1, s2, 0b0000000),
            SRL { d, s1, s2 }   => I32::r(0b0110011, d, 0b101, s1, s2, 0b0000000),
            SRA { d, s1, s2 }   => I32::r(0b0110011, d, 0b101, s1, s2, 0b0100000),
            OR { d, s1, s2 }    => I32::r(0b0110011, d, 0b110, s1, s2, 0b0000000),
            AND { d, s1, s2 }   => I32::r(0b0110011, d, 0b111, s1, s2, 0b0000000),
            ECALL {}            => 0b00000000000000000000000001110011,
            EBREAK {}           => 0b00000000000100000000000001110011,
            FENCE { im }        => I32::i(0b0001111, ZERO, 0b000, ZERO, im),

            // M-extension instructions
            MUL { d, s1, s2 }    => I32::r(0b0110011, d, 0b000, s1, s2, 0b0000001),
            MULH { d, s1, s2 }   => I32::r(0b0110011, d, 0b001, s1, s2, 0b0000001),
            MULHSU { d, s1, s2 } => I32::r(0b0110011, d, 0b010, s1, s2, 0b0000001),
            MULHU { d, s1, s2 }  => I32::r(0b0110011, d, 0b011, s1, s2, 0b0000001),
            DIV { d, s1, s2 }    => I32::r(0b0110011, d, 0b100, s1, s2, 0b0000001),
            DIVU { d, s1, s2 }   => I32::r(0b0110011, d, 0b101, s1, s2, 0b0000001),
            REM { d, s1, s2 }    => I32::r(0b0110011, d, 0b110, s1, s2, 0b0000001),
            REMU { d, s1, s2 }   => I32::r(0b0110011, d, 0b111, s1, s2, 0b0000001),

            // A-extension Load-Reserved/Store-Conditional
            LR_W { d, s1, aqrl }        => I32::amo(0b0101111, d, 0b010, s1, ZERO, aqrl, 0b00010),
            SC_W { d, s1, s2, aqrl }    => I32::amo(0b0101111, d, 0b010, s1, s2, aqrl, 0b00011),

            // A-extension Atomic Memory Operations (32-bit)
            AMOADD_W { d, s1, s2, aqrl }  => I32::amo(0b0101111, d, 0b010, s1, s2, aqrl, 0b00000),
            AMOSWAP_W { d, s1, s2, aqrl } => I32::amo(0b0101111, d, 0b010, s1, s2, aqrl, 0b00001),
            AMOAND_W { d, s1, s2, aqrl }  => I32::amo(0b0101111, d, 0b010, s1, s2, aqrl, 0b01100),
            AMOOR_W { d, s1, s2, aqrl }   => I32::amo(0b0101111, d, 0b010, s1, s2, aqrl, 0b01000),
            AMOXOR_W { d, s1, s2, aqrl }  => I32::amo(0b0101111, d, 0b010, s1, s2, aqrl, 0b00100),
            AMOMAX_W { d, s1, s2, aqrl }  => I32::amo(0b0101111, d, 0b010, s1, s2, aqrl, 0b10100),
            AMOMIN_W { d, s1, s2, aqrl }  => I32::amo(0b0101111, d, 0b010, s1, s2, aqrl, 0b10000),
            AMOMAXU_W { d, s1, s2, aqrl } => I32::amo(0b0101111, d, 0b010, s1, s2, aqrl, 0b11100),
            AMOMINU_W { d, s1, s2, aqrl } => I32::amo(0b0101111, d, 0b010, s1, s2, aqrl, 0b11000),
        }
    }

    #[inline(always)]
    #[allow(clippy::match_single_binding)]
    pub const fn try_from_u32(with: u32) -> Result<Self, ConversionError> {
        Self::try_from_u32_rv32(with)
    }

    #[inline(always)]
    #[allow(clippy::match_single_binding)]
    pub const fn try_from_u32_rv32(with: u32) -> Result<Self, ConversionError> {
        Self::try_from_u32_(with, 32)
    }

    #[inline(always)]
    #[allow(clippy::match_single_binding)]
    pub const fn try_from_u32_rv64(with: u32) -> Result<Self, ConversionError> {
        Self::try_from_u32_(with, 64)
    }

    #[allow(clippy::match_single_binding)]
    const fn try_from_u32_(with: u32, xlen: u8) -> Result<Self, ConversionError> {
        Ok(match with & 0b1111111 {
            // Load From RAM
            0b0000011 => match I32::from_i(with) {
                (d, 0b000, s, im) => LB { d, s, im },
                (d, 0b001, s, im) => LH { d, s, im },
                (d, 0b010, s, im) => LW { d, s, im },
                (d, 0b100, s, im) => LBU { d, s, im },
                (d, 0b101, s, im) => LHU { d, s, im },
                (_, funct, _, _) => {
                    return Err(ConversionError::UnknownFunct3(funct))
                }
            },
            // Misc. Memory Instructions
            0b0001111 => match I32::from_i(with) {
                (_, 0b000, _, im) => FENCE { im },
                (_, funct, _, _) => {
                    return Err(ConversionError::UnknownFunct3(funct))
                }
            },
            // Immediate Operations
            0b0010011 => match I32::from_i(with) {
                (d, 0b000, s, im) => ADDI { d, s, im },
                (d, 0b010, s, im) => SLTI { d, s, im },
                (d, 0b011, s, im) => SLTIU { d, s, im },
                (d, 0b100, s, im) => XORI { d, s, im },
                (d, 0b110, s, im) => ORI { d, s, im },
                (d, 0b111, s, im) => ANDI { d, s, im },

                (d, 0b001, s, im) => {
                    let shamt_mask = if xlen == 32 { 0x1F } else { 0x3F };
                    let shamt = (im & shamt_mask) as u8;
                    SLLI { d, s, shamt }
                },

                (d, 0b101, s, im) => {
                    let shamt_mask = if xlen == 32 { 0x1F } else { 0x3F };
                    let shamt = (im & shamt_mask) as u8;
                    let funct7 = (im >> 5) & 0b1111111;
                    match funct7 {
                        0b0000000 => SRLI { d, s, shamt },
                        0b0100000 => SRAI { d, s, shamt },
                        _ => return Err(ConversionError::UnknownFunct7(funct7 as _))
                    }
                },

                (_, funct3, _, _) => return Err(ConversionError::UnknownFunct3(funct3)),
            },
            // AUIPC
            0b0010111 => match I32::from_u(with) {
                (d, im) => AUIPC { d, im }
            },
            // Store To RAM
            0b0100011 => match I32::from_s(with) {
                (0b000, s1, s2, im) => SB { s1, s2, im },
                (0b001, s1, s2, im) => SH { s1, s2, im },
                (0b010, s1, s2, im) => SW { s1, s2, im },
                (funct, _, _, _) => {
                    return Err(ConversionError::UnknownFunct3(funct))
                }
            },
            // AMO (Atomic Memory Operations) - A-extension
            0b0101111 => match I32::from_amo(with) {
                (d, 0b010, s1, _, aqrl,  0b00010) => LR_W { d, s1, aqrl },
                (d, 0b010, s1, s2, aqrl, 0b00011) => SC_W { d, s1, s2, aqrl },
                (d, 0b010, s1, s2, aqrl, 0b00000) => AMOADD_W { d, s1, s2, aqrl },
                (d, 0b010, s1, s2, aqrl, 0b00001) => AMOSWAP_W { d, s1, s2, aqrl },
                (d, 0b010, s1, s2, aqrl, 0b01100) => AMOAND_W { d, s1, s2, aqrl },
                (d, 0b010, s1, s2, aqrl, 0b01000) => AMOOR_W { d, s1, s2, aqrl },
                (d, 0b010, s1, s2, aqrl, 0b00100) => AMOXOR_W { d, s1, s2, aqrl },
                (d, 0b010, s1, s2, aqrl, 0b10100) => AMOMAX_W { d, s1, s2, aqrl },
                (d, 0b010, s1, s2, aqrl, 0b10000) => AMOMIN_W { d, s1, s2, aqrl },
                (d, 0b010, s1, s2, aqrl, 0b11100) => AMOMAXU_W { d, s1, s2, aqrl },
                (d, 0b010, s1, s2, aqrl, 0b11000) => AMOMINU_W { d, s1, s2, aqrl },
                (_, funct3, _, _, _, funct5) => {
                    return Err(ConversionError::UnknownFunct3Funct5(funct3, funct5))
                }
            },
            // Register Operations
            0b0110011 => match I32::from_r(with) {
                (d, 0b000, s1, s2, 0b0000000) => ADD { d, s1, s2 },
                (d, 0b000, s1, s2, 0b0100000) => SUB { d, s1, s2 },
                (d, 0b001, s1, s2, 0b0000000) => SLL { d, s1, s2 },
                (d, 0b010, s1, s2, 0b0000000) => SLT { d, s1, s2 },
                (d, 0b011, s1, s2, 0b0000000) => SLTU { d, s1, s2 },
                (d, 0b100, s1, s2, 0b0000000) => XOR { d, s1, s2 },
                (d, 0b101, s1, s2, 0b0000000) => SRL { d, s1, s2 },
                (d, 0b101, s1, s2, 0b0100000) => SRA { d, s1, s2 },
                (d, 0b110, s1, s2, 0b0000000) => OR { d, s1, s2 },
                (d, 0b111, s1, s2, 0b0000000) => AND { d, s1, s2 },
                // M-extension instructions
                (d, 0b000, s1, s2, 0b0000001) => MUL { d, s1, s2 },
                (d, 0b001, s1, s2, 0b0000001) => MULH { d, s1, s2 },
                (d, 0b010, s1, s2, 0b0000001) => MULHSU { d, s1, s2 },
                (d, 0b011, s1, s2, 0b0000001) => MULHU { d, s1, s2 },
                (d, 0b100, s1, s2, 0b0000001) => DIV { d, s1, s2 },
                (d, 0b101, s1, s2, 0b0000001) => DIVU { d, s1, s2 },
                (d, 0b110, s1, s2, 0b0000001) => REM { d, s1, s2 },
                (d, 0b111, s1, s2, 0b0000001) => REMU { d, s1, s2 },
                (_, funct3, _, _, funct7) => {
                    return Err(ConversionError::UnknownFunct3Funct7(funct3, funct7))
                }
            },
            // LUI
            0b0110111 => match I32::from_u(with) {
                (d, im) => LUI { d, im }
            },
            // Branch Instructions
            0b1100011 => match I32::from_s(with) {
                (0b000, s1, s2, im) => BEQ { s1, s2, im },
                (0b001, s1, s2, im) => BNE { s1, s2, im },
                (0b100, s1, s2, im) => BLT { s1, s2, im },
                (0b101, s1, s2, im) => BGE { s1, s2, im },
                (0b110, s1, s2, im) => BLTU { s1, s2, im },
                (0b111, s1, s2, im) => BGEU { s1, s2, im },
                (funct, _, _, _) => {
                    return Err(ConversionError::UnknownFunct3(funct))
                }
            },
            // JALR
            0b1100111 => match I32::from_i(with) {
                (d, 0b000, s, im) => JALR { d, s, im },
                (_, funct, _, _) => {
                    return Err(ConversionError::UnknownFunct3(funct))
                }
            },
            // JAL
            0b1101111 => match I32::from_u(with) {
                (d, im) => JAL { d, im },
            },
            // System Instructions
            0b1110011 => match with {
                0b00000000000000000000000001110011 => ECALL {},
                0b00000000000100000000000001110011 => EBREAK {},
                _ => return Err(ConversionError::UnknownEnvCtrlTransfer)
            },
            _ => return Err(ConversionError::UnknownOpcode(with & 0b1111111))
        })
    }

}

impl From<I32> for u32 {
    #[inline(always)]
    fn from(with: I32) -> Self {
        I32::into_u32(with)
    }
}

impl TryFrom<u32> for I32 {
    type Error = ConversionError;
    // Using match makes it easier to extend code in the future.
    #[allow(clippy::match_single_binding)]
    fn try_from(with: u32) -> Result<Self, Self::Error> {
        Self::try_from_u32(with)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_b_type_roundtrip() {
        // test positive offset
        let encoded = I32::b(0b1100011, 0b000, RA, SP, 0x100);
        let (funct3, s1, s2, offset) = I32::from_b(encoded);
        assert_eq!(funct3, 0b000);
        assert_eq!(s1, RA);
        assert_eq!(s2, SP);
        assert_eq!(offset, 0x100);

        // Test negative offset
        let encoded = I32::b(0b1100011, 0b000, RA, SP, -256);
        let (.., offset) = I32::from_b(encoded);
        assert_eq!(offset, -256);

        // test edge case: maximum positive 13-bit offset (4094, since LSB must be 0)
        let encoded = I32::b(0b1100011, 0b001, T0, T1, 4094);
        let (funct3, s1, s2, offset) = I32::from_b(encoded);
        assert_eq!(funct3, 0b001);
        assert_eq!(s1, T0);
        assert_eq!(s2, T1);
        assert_eq!(offset, 4094);

        // test edge case: maximum negative 13-bit offset (-4096)
        let encoded = I32::b(0b1100011, 0b111, A0, A1, -4096);
        let (funct3, s1, s2, offset) = I32::from_b(encoded);
        assert_eq!(funct3, 0b111);
        assert_eq!(s1, A0);
        assert_eq!(s2, A1);
        assert_eq!(offset, -4096);
    }

    #[test]
    fn test_j_type_roundtrip() {
        // test simple cases first
        let test_cases = [
            0,
            2,       // minimum non-zero (LSB must be 0)
            0x800,   // bit 11 set
            0x1000,  // bit 12 set
            0x2000,  // bit 13 set
            -2,      // simple negative
            -0x800,  // negative with bit 11
            -0x1000, // negative with bit 12
        ];

        for &offset in &test_cases {
            let encoded = I32::j(0b1101111, RA, offset);
            let (d, decoded_offset) = I32::from_j(encoded);
            assert_eq!(d, RA);
            assert_eq!(decoded_offset, offset, "Failed for offset {:#x} ({})", offset, offset);
        }

        // test edge case: maximum positive 21-bit offset (1048574, since LSB must be 0)
        let encoded = I32::j(0b1101111, T6, 1048574);
        let (d, offset) = I32::from_j(encoded);
        assert_eq!(d, T6);
        assert_eq!(offset, 1048574);

        // test edge case: maximum negative 21-bit offset (-1048576)
        let encoded = I32::j(0b1101111, S11, -1048576);
        let (d, offset) = I32::from_j(encoded);
        assert_eq!(d, S11);
        assert_eq!(offset, -1048576);
    }

    #[test]
    fn test_r_type_roundtrip() {
        // test R-type instructions (ADD, SUB, etc.)
        let test_cases = [
            (T0, T1, T2),
            (ZERO, RA, SP),
            (A0, A1, S0),
            (T6, S11, GP),
        ];

        for &(rd, rs1, rs2) in &test_cases {
            // test ADD (funct7=0b0000000, funct3=0b000)
            let encoded = I32::r(0b0110011, rd, 0b000, rs1, rs2, 0b0000000);
            let (d, funct3, s1, s2, funct7) = I32::from_r(encoded);
            assert_eq!(d, rd);
            assert_eq!(funct3, 0b000);
            assert_eq!(s1, rs1);
            assert_eq!(s2, rs2);
            assert_eq!(funct7, 0b0000000);

            // test SUB (funct7=0b0100000, funct3=0b000)
            let encoded = I32::r(0b0110011, rd, 0b000, rs1, rs2, 0b0100000);
            let (d, funct3, s1, s2, funct7) = I32::from_r(encoded);
            assert_eq!(d, rd);
            assert_eq!(funct3, 0b000);
            assert_eq!(s1, rs1);
            assert_eq!(s2, rs2);
            assert_eq!(funct7, 0b0100000);
        }
    }

    #[test]
    fn test_i_type_roundtrip() {
        // test I-type instructions (ADDI, LW, etc.)
        let test_cases = [
            (T0, T1, 0i16),
            (A0, SP, 100i16),
            (S0, RA, -50i16),
            (T6, ZERO, 2047i16), // max positive 12-bit
            (GP, T2, -2048i16),  // max negative 12-bit
        ];

        for &(rd, rs1, imm) in &test_cases {
            // test ADDI (funct3=0b000)
            let encoded = I32::i(0b0010011, rd, 0b000, rs1, imm);
            let (d, funct3, s1, decoded_imm) = I32::from_i(encoded);
            assert_eq!(d, rd);
            assert_eq!(funct3, 0b000);
            assert_eq!(s1, rs1);
            assert_eq!(decoded_imm, imm);

            // test LW (funct3=0b010)
            let encoded = I32::i(0b0000011, rd, 0b010, rs1, imm);
            let (d, funct3, s1, decoded_imm) = I32::from_i(encoded);
            assert_eq!(d, rd);
            assert_eq!(funct3, 0b010);
            assert_eq!(s1, rs1);
            assert_eq!(decoded_imm, imm);
        }
    }

    #[test]
    fn test_i7_type_roundtrip() {
        // test I7-type instructions (SLLI, SRLI, SRAI)
        let test_cases = [
            (T0, T1, 0i8),
            (A0, SP, 15i8),
            (S0, RA, 31i8), // max shift amount
            (T6, ZERO, 1i8),
        ];

        for &(rd, rs1, shamt) in &test_cases {
            // test SLLI (funct7=0b0000000, funct3=0b001)
            let encoded = I32::i7(0b0010011, rd, 0b001, rs1, shamt, 0b0000000);
            let (d, funct3, s1, decoded_shamt, funct7) = I32::from_i7(encoded);
            assert_eq!(d, rd);
            assert_eq!(funct3, 0b001);
            assert_eq!(s1, rs1);
            assert_eq!(decoded_shamt, shamt);
            assert_eq!(funct7, 0b0000000);

            // test SRAI (funct7=0b0100000, funct3=0b101)
            let encoded = I32::i7(0b0010011, rd, 0b101, rs1, shamt, 0b0100000);
            let (d, funct3, s1, decoded_shamt, funct7) = I32::from_i7(encoded);
            assert_eq!(d, rd);
            assert_eq!(funct3, 0b101);
            assert_eq!(s1, rs1);
            assert_eq!(decoded_shamt, shamt);
            assert_eq!(funct7, 0b0100000);
        }
    }

    #[test]
    fn test_s_type_roundtrip() {
        // test S-type instructions (SW, SH, SB)
        let test_cases = [
            (T1, T2, 0i16),
            (SP, A0, 100i16),
            (RA, S0, -50i16),
            (ZERO, T6, 2047i16),    // max positive 12-bit
            (T2, GP, -2048i16),     // max negative 12-bit
        ];

        for &(rs1, rs2, imm) in &test_cases {
            // test SW (funct3=0b010)
            let encoded = I32::s(0b0100011, 0b010, rs1, rs2, imm);
            let (funct3, s1, s2, decoded_imm) = I32::from_s(encoded);
            assert_eq!(funct3, 0b010);
            assert_eq!(s1, rs1);
            assert_eq!(s2, rs2);
            assert_eq!(decoded_imm, imm);

            // test SB (funct3=0b000)
            let encoded = I32::s(0b0100011, 0b000, rs1, rs2, imm);
            let (funct3, s1, s2, decoded_imm) = I32::from_s(encoded);
            assert_eq!(funct3, 0b000);
            assert_eq!(s1, rs1);
            assert_eq!(s2, rs2);
            assert_eq!(decoded_imm, imm);
        }
    }

    #[test]
    fn test_sign_extension_u_type() {
        // test positive values
        let encoded = I32::u(0b0110111, T0, 0x12345);
        let (_, decoded) = I32::from_u(encoded);
        assert_eq!(decoded, 0x12345);

        // test negative values (LUI/AUIPC usage often uses signed semantics)
        let encoded = I32::u(0b0110111, T0, -1);
        let (_, decoded) = I32::from_u(encoded);
        assert_eq!(decoded, -1);

        // test positive boundary (max positive signed 20-bit)
        let encoded = I32::u(0b0110111, T0, 0x7FFFF); // 524_287
        let (_, decoded) = I32::from_u(encoded);
        assert_eq!(decoded, 0x7FFFF);

        // test sign bit boundary (bit 19 set => smallest negative)
        let encoded = I32::u(0b0110111, T0, -524_288); // -2^19
        let (_, decoded) = I32::from_u(encoded);
        assert_eq!(decoded, -524_288);
    }

    #[test]
    fn test_edge_cases() {
        // test register boundary cases
        let all_regs = [
            ZERO, RA, SP, GP, TP,
            T0, T1, T2, S0, S1,
            A0, A1, A2, A3, A4,
            A5, A6, A7, S2, S3,
            S4, S5, S6, S7, S8,
            S9, S10, S11, T3, T4,
            T5, T6,
        ];

        // test that all 32 registers encode/decode correctly
        for (i, &reg) in all_regs.iter().enumerate() {
            let encoded = I32::r(0b0110011, reg, 0b000, ZERO, ZERO, 0b0000000);
            let (d, .., _) = I32::from_r(encoded);
            assert_eq!(d, reg);
            assert_eq!(reg as u32, i as u32);
        }

        // test immediate value boundaries for I-type
        let i_boundary_tests = [
            2047i16,   // max positive 12-bit signed
            -2048i16,  // max negative 12-bit signed
            0i16,      // zero
            1i16,      // small positive
            -1i16,     // small negative
        ];

        for &imm in &i_boundary_tests {
            let encoded = I32::i(0b0010011, T0, 0b000, T1, imm);
            let (.., decoded_imm) = I32::from_i(encoded);
            assert_eq!(decoded_imm, imm);
        }
    }

    #[test]
    fn debug_j_type_no_std() {
        let offset = 0x1000i32; // 4096

        let encoded = I32::j(0b1101111, RA, offset);

        // let's manually check the bit fields
        let imm_bits_31 = (encoded >> 31) & 0x1;
        let imm_bits_30_21 = (encoded >> 21) & 0x3FF;
        let imm_bit_20 = (encoded >> 20) & 0x1;
        let imm_bits_19_12 = (encoded >> 12) & 0xFF;

        let (d, decoded_offset) = I32::from_j(encoded);

        assert_eq!(d, RA);

        // for 0x1000, we expect:
        // - imm[12] = 1 (bit 12 of 0x1000)
        // - All other bits = 0
        // so imm[19:12] should be 0x01
        assert_eq!(imm_bits_19_12, 0x01); // bit 12 of original -> bit 0 of this field
        assert_eq!(imm_bits_30_21, 0);    // imm[10:1] should be 0
        assert_eq!(imm_bit_20, 0);        // imm[11] should be 0
        assert_eq!(imm_bits_31, 0);       // imm[20] should be 0 (positive number)

        assert_eq!(decoded_offset, offset);
    }

    #[test]
    fn test_sign_extension_i_type() {
        // test positive values
        let encoded = I32::i(0b0010011, T0, 0b000, T1, 100);
        let (.., decoded) = I32::from_i(encoded);
        assert_eq!(decoded, 100);

        // test negative values
        let encoded = I32::i(0b0010011, T0, 0b000, T1, -50);
        let (.., decoded) = I32::from_i(encoded);
        assert_eq!(decoded, -50);

        // test boundary values
        let encoded = I32::i(0b0010011, T0, 0b000, T1, 2047);  // max positive 12-bit
        let (.., decoded) = I32::from_i(encoded);
        assert_eq!(decoded, 2047);

        let encoded = I32::i(0b0010011, T0, 0b000, T1, -2048); // max negative 12-bit
        let (.., decoded) = I32::from_i(encoded);
        assert_eq!(decoded, -2048);
    }

    #[test]
    fn test_sign_extension_s_type() {
        // test positive values
        let encoded = I32::s(0b0100011, 0b010, T0, T1, 100);
        let (.., decoded) = I32::from_s(encoded);
        assert_eq!(decoded, 100);

        // test negative values
        let encoded = I32::s(0b0100011, 0b010, T0, T1, -50);
        let (.., decoded) = I32::from_s(encoded);
        assert_eq!(decoded, -50);

        // test boundary values
        let encoded = I32::s(0b0100011, 0b010, T0, T1, 2047);
        let (.., decoded) = I32::from_s(encoded);
        assert_eq!(decoded, 2047);

        let encoded = I32::s(0b0100011, 0b010, T0, T1, -2048);
        let (.., decoded) = I32::from_s(encoded);
        assert_eq!(decoded, -2048);
    }

    #[test]
    fn test_u_type_roundtrip() {
        // test U-type instructions (LUI, AUIPC) with values in signed 20-bit range
        let test_cases = [
            (T0, 0i32),
            (A0, 0x12345i32),
            (SP, -1i32),       // all bits set -> -1
            (RA, -524_288i32), // bit 19 set => -524_288
            (T6, 0x7FFFFi32),  // max positive signed 20-bit value = 524_287
        ];

        for &(rd, imm) in &test_cases {
            // test LUI
            let encoded = I32::u(0b0110111, rd, imm);
            let (d, decoded_imm) = I32::from_u(encoded);
            assert_eq!(d, rd);
            assert_eq!(decoded_imm, imm);

            // test AUIPC
            let encoded = I32::u(0b0010111, rd, imm);
            let (d, decoded_imm) = I32::from_u(encoded);
            assert_eq!(d, rd);
            assert_eq!(decoded_imm, imm);
        }
    }

    #[test]
    fn test_bit_patterns() {
        // verify that -1 as i16 becomes 0xFFFF when cast to u16
        assert_eq!((-1i16) as u16, 0xFFFF);
        assert_eq!((-50i16) as u16, 0xFFCE);

        // verify our sign extension logic
        let test_val = 0xFFCE_u16; // -50 as u16
        let sign_extended = if (test_val & 0x800) != 0 {
            (test_val | 0xF000) as i16
        } else {
            test_val as i16
        };
        assert_eq!(sign_extended, -50i16);
    }
}
