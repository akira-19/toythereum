use std::{
    collections::HashMap,
    env,
    error::Error,
    fmt::Display,
    fs::File,
    hash::Hash,
    io::{BufReader, BufWriter, Read, Write},
};

use primitive_types::U256;
use tiny_keccak::{keccakf, Hasher, Keccak};

use crate::parser::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpCode {
    Stop = 0x00,
    Add = 0x01,
    Mul = 0x02,
    Sub = 0x03,
    Div = 0x04,

    EQ = 0x14,
    SHR = 0x1C,

    CalldataLoad = 0x35,

    MLoad = 0x51,
    MStore = 0x52,
    SLoad = 0x54,
    SStore = 0x55,
    Jump = 0x56,
    JumpI = 0x57,
    JumpDest = 0x5B,

    Push1 = 0x60,
    Push32 = 0x7F,

    Return = 0xF3,
}

macro_rules! impl_op_from {
    ($($op:ident),*) => {
      impl From<u8> for OpCode {
        #[allow(non_upper_case_globals)]
        fn from(o: u8) -> Self {
          $(const $op: u8 = OpCode::$op as u8;)*

          match o {
            $($op => Self::$op,)*
            _ => panic!("Opcode \"{:02X}\" unrecognized!", o),
          }
        }
      }
    }
  }

impl_op_from!(Stop, Add, MStore, Push1, Return);

#[derive(Debug, Clone, Copy)]
enum ArgValue {
    U256(U256),
    FnSelector([u8; 4]),
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Instruction {
    op: OpCode,
    arg0: Option<ArgValue>,
}

impl Instruction {
    fn new(op: OpCode, arg0: Option<ArgValue>) -> Self {
        Self { op, arg0 }
    }

    // fn serialize(&self, writer: &mut impl Write) -> Result<(), std::io::Error> {
    //     writer.write_all(&[self.op as u8, self.arg0])?;
    //     Ok(())
    // }

    // fn deserialize(reader: &mut impl Read) -> Result<Self, std::io::Error> {
    //     let mut buf = [0u8; 2];
    //     reader.read_exact(&mut buf)?;
    //     Ok(Self::new(buf[0].into(), buf[1]))
    // }
}

fn serialize_size(sz: usize, writer: &mut impl Write) -> std::io::Result<()> {
    writer.write_all(&(sz as u32).to_le_bytes())
}

fn deserialize_size(reader: &mut impl Read) -> std::io::Result<usize> {
    let mut buf = [0u8; std::mem::size_of::<u32>()];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf) as usize)
}

pub fn serialize_str(s: &str, writer: &mut impl Write) -> std::io::Result<()> {
    serialize_size(s.len(), writer)?;
    writer.write_all(s.as_bytes())?;
    Ok(())
}

pub fn deserialize_str(reader: &mut impl Read) -> std::io::Result<String> {
    let mut buf = vec![0u8; deserialize_size(reader)?];
    reader.read_exact(&mut buf)?;
    let s = String::from_utf8(buf).unwrap();
    Ok(s)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// Absolute Stack Index
struct StkIdx(usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// Instruction Pointer
struct InstPtr(usize);

struct FnByteCode {
    args: Vec<String>,
    literals: Vec<Value>,
    instructions: Vec<Instruction>,
}

impl FnByteCode {
    fn write_args(args: &[String], writer: &mut impl Write) -> std::io::Result<()> {
        serialize_size(args.len(), writer)?;
        for arg in args {
            serialize_str(arg, writer)?;
        }
        Ok(())
    }

    fn write_literals(literals: &[Value], writer: &mut impl Write) -> std::io::Result<()> {
        serialize_size(literals.len(), writer)?;
        for value in literals {
            value.serialize(writer)?;
        }
        Ok(())
    }

    // fn write_insts(instructions: &[Instruction], writer: &mut impl Write) -> std::io::Result<()> {
    //     serialize_size(instructions.len(), writer)?;
    //     for instruction in instructions {
    //         instruction.serialize(writer).unwrap();
    //     }
    //     Ok(())
    // }

    // fn serialize(&self, writer: &mut impl Write) -> std::io::Result<()> {
    //     Self::write_args(&self.args, writer)?;
    //     Self::write_literals(&self.literals, writer)?;
    //     Self::write_insts(&self.instructions, writer)?;
    //     Ok(())
    // }

    fn read_args(reader: &mut impl Read) -> std::io::Result<Vec<String>> {
        let num_args = deserialize_size(reader)?;
        let mut args = Vec::with_capacity(num_args);
        for _ in 0..num_args {
            args.push(deserialize_str(reader)?);
        }
        Ok(args)
    }

    fn read_literals(reader: &mut impl Read) -> std::io::Result<Vec<Value>> {
        let num_literals = deserialize_size(reader)?;
        let mut literals = Vec::with_capacity(num_literals);
        for _ in 0..num_literals {
            literals.push(Value::deserialize(reader)?);
        }
        Ok(literals)
    }

    // fn read_instructions(reader: &mut impl Read) -> std::io::Result<Vec<Instruction>> {
    //     let num_instructions = deserialize_size(reader)?;
    //     let mut instructions = Vec::with_capacity(num_instructions);
    //     for _ in 0..num_instructions {
    //         let inst = Instruction::deserialize(reader)?;
    //         instructions.push(inst);
    //     }
    //     Ok(instructions)
    // }

    // fn deserialize(reader: &mut impl Read) -> std::io::Result<Self> {
    //     let args = Self::read_args(reader)?;
    //     let literals = Self::read_literals(reader)?;
    //     let instructions = Self::read_instructions(reader)?;
    //     Ok(Self {
    //         args,
    //         literals,
    //         instructions,
    //     })
    // }

    fn disasm(&self, writer: &mut impl Write) -> std::io::Result<()> {
        disasm_common(&self.literals, &self.instructions, writer)
    }
}

fn disasm_common(
    literals: &[Value],
    instructions: &[Instruction],
    writer: &mut impl Write,
) -> std::io::Result<()> {
    use OpCode::*;
    writeln!(writer, "  Literals [{}]", literals.len())?;
    for (i, con) in literals.iter().enumerate() {
        writeln!(writer, "    [{i}] {}", *con)?;
    }

    writeln!(writer, "  Instructions [{}]", instructions.len())?;
    // for (i, inst) in instructions.iter().enumerate() {
    //     match inst.op {
    //         LoadLiteral => writeln!(
    //             writer,
    //             "    [{i}] {:?} {} ({:?})",
    //             inst.op, inst.arg0, literals[inst.arg0 as usize]
    //         )?,
    //         Copy | Dup | Call | Jmp | Jf | Pop | Store | Ret => {
    //             writeln!(writer, "    [{i}] {:?} {}", inst.op, inst.arg0)?
    //         }
    //         _ => writeln!(writer, "    [{i}] {:?}", inst.op)?,
    //     }
    // }
    Ok(())
}

#[derive(Debug, Clone)]
struct MemoryVariableTable {
    value: Value,
    address: U256,
}

#[derive(Debug, Clone)]
struct StorageVariableTable {
    value: Value,
    slot: U256,
}

#[derive(Debug, Clone)]
struct Arg {
    name: String,
    value_type: TypeDecl,
}

#[derive(Debug, Clone)]
struct FnStatements<'a> {
    statements: Vec<Statement<'a>>,
    args: Vec<Arg>,
}

struct Compiler<'a> {
    literals: Vec<Value>,
    instructions: Vec<Instruction>,
    funcs: HashMap<[u8; 4], FnStatements<'a>>,
    memories: HashMap<String, MemoryVariableTable>,
    memory_pointer: U256,
    storages: HashMap<String, StorageVariableTable>,
    slot: U256,
    pc: usize,
}

impl<'a> Compiler<'a> {
    fn new() -> Self {
        Self {
            literals: vec![],
            instructions: vec![],
            funcs: HashMap::new(),
            memories: HashMap::new(),
            memory_pointer: U256::zero(),
            storages: HashMap::new(),
            slot: U256::zero(),
            pc: 0,
        }
    }

    // fn stack_top(&self) -> StkIdx {
    //     StkIdx(self.target_stack.len() - 1)
    // }

    // fn add_literal(&mut self, value: Value) -> u8 {
    //     let existing = self
    //         .literals
    //         .iter()
    //         .enumerate()
    //         .find(|(_, val)| **val == value);
    //     if let Some((i, _)) = existing {
    //         i as u8
    //     } else {
    //         let ret = self.literals.len();
    //         self.literals.push(value);
    //         ret as u8
    //     }
    // }

    /// Returns absolute position of inserted value
    fn add_inst(&mut self, op: OpCode, arg0: Option<ArgValue>) -> InstPtr {
        let inst = self.instructions.len();
        self.instructions.push(Instruction { op, arg0 });
        self.pc += 1;
        if arg0.is_some() {
            self.pc += 32;
        }
        InstPtr(inst)
    }

    fn compile_expr(&mut self, ex: &Expression) -> Result<Value, Box<dyn Error>> {
        Ok(match &ex.expr {
            ExprEnum::NumLiteral(num) => {
                self.add_inst(OpCode::Push32, Some(ArgValue::U256(U256::from(*num))));
                return Ok(Value::Uint256(U256::from(*num)));
            }
            // ExprEnum::StrLiteral(str) => {
            //     let id = self.add_literal(Value::Str(str.clone()));
            //     self.add_load_literal_inst(id);
            //     self.stack_top()
            // }
            // ExprEnum::BoolLiteral(b) => {
            //     let id = self.add_literal(Value::Bool(*b));
            //     self.add_load_literal_inst(id);
            //     self.stack_top()
            // }
            ExprEnum::Ident(ident) => {
                if let Some(v) = self.memories.get(&ident.fragment().to_string()).cloned() {
                    self.load_memory(&v)?;
                    return Ok(v.value);
                } else if let Some(v) = self.storages.get(&ident.fragment().to_string()).cloned() {
                    self.load_storage(&v)?;
                    return Ok(v.value);
                } else {
                    return Err(format!("Variable not found: {ident:?}").into());
                };
            }
            ExprEnum::Add(lhs, rhs) => self.bin_op(OpCode::Add, lhs, rhs)?,
            ExprEnum::Sub(lhs, rhs) => self.bin_op(OpCode::Sub, lhs, rhs)?,
            ExprEnum::Mul(lhs, rhs) => self.bin_op(OpCode::Mul, lhs, rhs)?,
            ExprEnum::Div(lhs, rhs) => self.bin_op(OpCode::Div, lhs, rhs)?,
            // ExprEnum::Gt(lhs, rhs) => self.bin_op(OpCode::Lt, rhs, lhs)?,
            // ExprEnum::Lt(lhs, rhs) => self.bin_op(OpCode::Lt, lhs, rhs)?,
            // ExprEnum::FnInvoke(name, args) => {
            //     let stack_before_args = self.target_stack.len();
            //     let name = self.add_literal(Value::Str(name.to_string()));
            //     let args = args
            //         .iter()
            //         .map(|arg| self.compile_expr(arg))
            //         .collect::<Result<Vec<_>, _>>()?;

            //     let stack_before_call = self.target_stack.len();
            //     self.add_load_literal_inst(name);
            //     for arg in &args {
            //         self.add_copy_inst(*arg);
            //     }

            //     self.add_inst(OpCode::Call, args.len() as u8);
            //     self.target_stack
            //         .resize(stack_before_call + 1, Target::Temp);
            //     self.coerce_stack(StkIdx(stack_before_args));
            //     self.stack_top()
            // }
            // ExprEnum::If(cond, true_branch, false_branch) => {
            //     use OpCode::*;
            //     let cond = self.compile_expr(cond)?;
            //     self.add_copy_inst(cond);
            //     let jf_inst = self.add_jf_inst();
            //     let stack_size_before = self.target_stack.len();
            //     self.compile_stmts_or_zero(true_branch, false)?;
            //     self.coerce_stack(StkIdx(stack_size_before + 1));
            //     let jmp_inst = self.add_inst(Jmp, 0);
            //     self.fixup_jmp(jf_inst);
            //     self.target_stack.resize(stack_size_before, Target::Temp);
            //     if let Some(false_branch) = false_branch.as_ref() {
            //         self.compile_stmts_or_zero(&false_branch, false)?;
            //     }
            //     self.coerce_stack(StkIdx(stack_size_before + 1));
            //     self.fixup_jmp(jmp_inst);
            //     self.stack_top()
            // }
            _ => todo!(),
        })
    }

    fn load_memory(&mut self, v: &MemoryVariableTable) -> Result<(), Box<dyn Error>> {
        // TODO: if the value is dynamic type, we need to load the memory recursively?
        self.add_inst(OpCode::Push32, Some(ArgValue::U256(v.address)));
        self.add_inst(OpCode::MLoad, None);
        Ok(())
    }

    fn save_memory(&mut self, name: String, v: &Value) -> Result<(), Box<dyn Error>> {
        // TODO: if the value is dynamic type, we need to user more than one memory slot
        // but for now, we only use one slot

        self.memories.insert(
            name,
            MemoryVariableTable {
                value: v.clone(),
                address: self.memory_pointer,
            },
        );

        let value = v.to_vec_u8();
        self.add_inst(
            OpCode::Push32,
            Some(ArgValue::U256(U256::from_big_endian(&value))),
        );
        self.add_inst(OpCode::Push32, Some(ArgValue::U256(self.memory_pointer)));
        self.add_inst(OpCode::MStore, None);
        self.memory_pointer += U256::from(32);
        Ok(())
    }

    fn update_memory(&mut self, name: String, v: &Value) -> Result<(), Box<dyn Error>> {
        if let Some(m) = self.memories.get_mut(&name) {
            let address = m.address;
            m.value = v.clone();
            let value = v.to_vec_u8();

            // TODO: if the value is dynamic type, new memory slots might be needed
            self.add_inst(
                OpCode::Push32,
                Some(ArgValue::U256(U256::from_big_endian(&value))),
            );
            self.add_inst(OpCode::Push32, Some(ArgValue::U256(address)));
            self.add_inst(OpCode::MStore, None);
            Ok(())
        } else {
            Err(format!("Variable not found: {name:?}").into())
        }
    }

    fn load_storage(&mut self, v: &StorageVariableTable) -> Result<(), Box<dyn Error>> {
        self.add_inst(OpCode::Push32, Some(ArgValue::U256(v.slot)));
        match &v.value {
            Value::Uint256(_) => {
                self.add_inst(OpCode::SLoad, None);
            }
            Value::Str(_) => {
                // retrieve the length from the slot
                self.add_inst(OpCode::SLoad, None);

                // get pointer to the slot of the string
                let mut hasher = Keccak::v256();
                let mut pointer = [0u8; 32];
                hasher.update(&v.slot.to_big_endian().to_vec());
                hasher.finalize(&mut pointer);
                self.add_inst(
                    OpCode::Push32,
                    Some(ArgValue::U256(U256::from_big_endian(&pointer))),
                );

                // load the string from the slot
                self.add_inst(OpCode::SLoad, None);

                // TODO: implement the process if the string is longer than 32 bytes
            }
            Value::Bool(_) => {
                self.add_inst(OpCode::SLoad, None);
            }
        }
        Ok(())
    }

    fn save_storage(&mut self, name: String, v: &Value) -> Result<(), Box<dyn Error>> {
        self.storages.insert(
            name,
            StorageVariableTable {
                value: v.clone(),
                slot: self.slot,
            },
        );
        match v {
            Value::Uint256(value) => {
                self.add_inst(OpCode::Push32, Some(ArgValue::U256(*value)));
                self.add_inst(OpCode::Push32, Some(ArgValue::U256(self.slot)));
                self.add_inst(OpCode::SStore, None);
                self.slot += U256::one();
            }
            Value::Str(value) => {
                // store the length of the string
                let len = v.to_vec_u8().len();
                self.add_inst(OpCode::Push32, Some(ArgValue::U256(U256::from(len))));
                self.add_inst(OpCode::Push32, Some(ArgValue::U256(self.slot)));
                self.add_inst(OpCode::SStore, None);
                self.slot += U256::one();

                // store the string
                // TODO: implement the process if the string is longer than 32 bytes
                // but for now, only one slot will be used
                let mut hasher = Keccak::v256();
                let mut pointer = [0u8; 32];
                hasher.update(&self.slot.to_big_endian().to_vec());
                hasher.finalize(&mut pointer);
                self.add_inst(
                    OpCode::Push32,
                    Some(ArgValue::U256(U256::from_big_endian(
                        &value.clone().into_bytes(),
                    ))),
                );
                self.add_inst(
                    OpCode::Push32,
                    Some(ArgValue::U256(U256::from_big_endian(&pointer))),
                );
                self.add_inst(OpCode::SStore, None);
            }
            Value::Bool(value) => {
                let num = if *value { U256::one() } else { U256::zero() };
                self.add_inst(OpCode::Push32, Some(ArgValue::U256(num)));
                self.add_inst(OpCode::Push32, Some(ArgValue::U256(self.slot)));
                self.add_inst(OpCode::SStore, None);
                self.slot += U256::one();
            }
        }
        Ok(())
    }

    fn update_storage(&mut self, name: String, v: &Value) -> Result<(), Box<dyn Error>> {
        if let Some(s) = self.storages.get_mut(&name) {
            s.value = v.clone();
            let slot = s.slot;
            match v {
                Value::Uint256(value) => {
                    self.add_inst(OpCode::Push32, Some(ArgValue::U256(*value)));
                    self.add_inst(OpCode::Push32, Some(ArgValue::U256(slot)));
                    self.add_inst(OpCode::SStore, None);
                }
                Value::Str(value) => {
                    // store the length of the string
                    let len = v.to_vec_u8().len();
                    self.add_inst(OpCode::Push32, Some(ArgValue::U256(U256::from(len))));
                    self.add_inst(OpCode::Push32, Some(ArgValue::U256(slot)));
                    self.add_inst(OpCode::SStore, None);

                    // store the string
                    // TODO: implement the process if the string is longer than 32 bytes
                    // but for now, only one slot will be used
                    let mut hasher = Keccak::v256();
                    let mut pointer = [0u8; 32];
                    hasher.update(&self.slot.to_big_endian().to_vec());
                    hasher.finalize(&mut pointer);
                    self.add_inst(
                        OpCode::Push32,
                        Some(ArgValue::U256(U256::from_big_endian(
                            &value.clone().into_bytes(),
                        ))),
                    );
                    self.add_inst(
                        OpCode::Push32,
                        Some(ArgValue::U256(U256::from_big_endian(&pointer))),
                    );
                    self.add_inst(OpCode::SStore, None);
                }
                Value::Bool(value) => {
                    let num = if *value { U256::one() } else { U256::zero() };
                    self.add_inst(OpCode::Push32, Some(ArgValue::U256(num)));
                    self.add_inst(OpCode::Push32, Some(ArgValue::U256(slot)));
                    self.add_inst(OpCode::SStore, None);
                }
            }
            Ok(())
        } else {
            Err(format!("Variable not found: {name:?}").into())
        }
    }

    fn bin_op(
        &mut self,
        op: OpCode,
        lhs: &Expression,
        rhs: &Expression,
    ) -> Result<Value, Box<dyn Error>> {
        let r_value = self.compile_expr(rhs)?;
        let l_value = self.compile_expr(lhs)?;

        self.add_inst(op, None);

        match (l_value, r_value) {
            (Value::Uint256(l), Value::Uint256(r)) => {
                let result = match op {
                    OpCode::Add => l + r,
                    OpCode::Sub => l - r,
                    OpCode::Mul => l * r,
                    OpCode::Div => {
                        if r == U256::zero() {
                            return Err("Division by zero".into());
                        }
                        l / r
                    }
                    _ => return Err("Invalid opcode".into()),
                };
                Ok(Value::Uint256(result))
            }
            _ => Err("Invalid type".into()),
        }
    }

    fn compile_stmts(
        &mut self,
        stmts: &'a Statements<'a>,
        top: bool,
    ) -> Result<Option<StkIdx>, Box<dyn Error>> {
        let mut last_result = None;
        for stmt in stmts {
            match stmt {
                Statement::Expression(ex) => {
                    last_result = Some(self.compile_expr(ex)?);
                }
                Statement::VarDef { name, ex, .. } => {
                    let result = self.compile_expr(ex)?;
                    let var_name = name.fragment().to_string();
                    // if top is true, we put the value into the storage
                    if top {
                        self.save_storage(var_name, &result)?;
                    } else {
                        self.save_memory(var_name, &result)?;
                    }
                }
                Statement::VarAssign { name, ex, .. } => {
                    let result = self.compile_expr(ex)?;
                    let var_name = name.fragment().to_string();

                    if let Some(data) = self.memories.get_mut(&var_name) {
                        data.value = result.clone();
                        let value = data.value.clone();
                        self.update_memory(var_name, &value)?;
                    } else if let Some(data) = self.storages.get_mut(&var_name) {
                        data.value = result.clone();
                        let value = data.value.clone();
                        self.update_storage(var_name, &value)?;
                    } else {
                        return Err(format!("Variable not found: {name:?}").into());
                    }
                }
                Statement::FnDef {
                    name, args, stmts, ..
                } => {
                    let func = name.fragment();
                    let arg_types = args.iter().map(|arg| arg.1.type_name()).collect::<Vec<_>>();
                    let selector = create_func_selector(func, arg_types);

                    // get func selector from calldata
                    self.add_inst(OpCode::CalldataLoad, None);
                    self.add_inst(
                        OpCode::Push32,
                        Some(ArgValue::U256(U256::from_str_radix("E0", 16).unwrap())),
                    );
                    self.add_inst(OpCode::SHR, None);

                    // add this function's selector to the stack and compare
                    self.add_inst(
                        OpCode::Push32,
                        Some(ArgValue::U256(U256::from_big_endian(&selector))),
                    );
                    self.add_inst(OpCode::EQ, None);

                    // if the selector is equal, jmp
                    self.add_inst(OpCode::Push32, Some(ArgValue::FnSelector(selector)));
                    self.add_inst(OpCode::JumpI, None);

                    // save the func temporarily
                    let fn_args = args
                        .iter()
                        .map(|arg| Arg {
                            name: arg.0.fragment().to_string(),
                            value_type: arg.1,
                        })
                        .collect::<Vec<_>>();

                    let fn_stmts = FnStatements {
                        statements: stmts.clone(),
                        args: fn_args,
                    };

                    self.funcs.insert(selector, fn_stmts);
                }
                Statement::Return(ex) => {
                    // let res = self.compile_expr(ex)?;
                    // self.add_inst(OpCode::Ret, (self.target_stack.len() - res.0 - 1) as u8);
                }
            }
        }
        for inst in &self.instructions {
            println!("{:?}", inst);
        }
        // put funcs here

        // func selector to pc

        Ok(None)
        // Ok(last_result)
    }

    fn compile(&mut self, stmts: &'a Statements<'a>) -> Result<(), Box<dyn std::error::Error>> {
        self.init_memory()?;
        self.compile_stmts(stmts, true)?;
        // compile the functions

        for (selector, fn_stmts) in self.funcs.clone().iter() {
            self.compile_func(*selector, fn_stmts)?;
        }
        Ok(())
    }

    fn compile_func<'b>(
        &mut self,
        selector: [u8; 4],
        fn_stmts: &'b FnStatements<'b>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // allocate the current pc
        self.instructions.iter_mut().for_each(|inst| {
            if inst.op == OpCode::Push32 {
                if let Some(ArgValue::FnSelector(fn_selector)) = inst.arg0 {
                    if fn_selector == selector {
                        // set the pc
                        inst.arg0 = Some(ArgValue::U256(U256::from(self.pc)));
                    }
                }
            }
        });

        // add jumpdest
        self.instructions
            .push(Instruction::new(OpCode::JumpDest, None));
        // put args into memory
        for arg in &fn_stmts.args {
            // self.save_memory(arg.name.clone(), &Value::Uint256(U256::zero()))?;
        }

        // self.compile_stmts(stmts, false)?;
        Ok(())
    }

    // fn extract_value_from_calldata(
    //     &mut self,
    //     offset: usize,
    //     arg: &Arg,
    // ) -> Result<Value, Box<dyn Error>> {
    //     self.add_inst(OpCode::Push32, Some(ArgValue::U256(U256::from(offset))));
    //     self.add_inst(OpCode::CalldataLoad, None);

    //     let value = match arg.value_type {
    //         TypeDecl::Uint256 => Value::Uint256(U256::from_big_endian(arg.)),
    //         TypeDecl::Str => {}
    //     };

    //     self.memories.insert(
    //         arg.name.clone(),
    //         MemoryVariableTable {
    //             value: v.clone(),
    //             address: self.memory_pointer,
    //         },
    //     );

    //     let value = v.to_vec_u8();
    //     self.add_inst(
    //         OpCode::Push32,
    //         Some(ArgValue::U256(U256::from_big_endian(&value))),
    //     );
    //     self.add_inst(OpCode::Push32, Some(ArgValue::U256(self.memory_pointer)));
    //     self.add_inst(OpCode::MStore, None);
    //     self.memory_pointer += U256::from(32);
    //     Ok(())
    // }

    fn init_memory(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.memory_pointer = U256::from_str_radix("80", 16)?;
        self.add_inst(OpCode::Push32, Some(ArgValue::U256(self.memory_pointer)));
        self.add_inst(
            OpCode::Push32,
            Some(ArgValue::U256(U256::from_str_radix("40", 16)?)),
        );
        self.add_inst(OpCode::MStore, None);
        Ok(())
    }
}

fn create_func_selector(name: &str, args: Vec<&str>) -> [u8; 4] {
    let mut func = name.to_string();
    func += "(";
    for arg in args {
        func += arg;
    }
    func += ")";

    let mut full_selector = [0u8; 32];
    let mut hasher = Keccak::v256();
    hasher.update(func.as_bytes());
    hasher.finalize(&mut full_selector);
    full_selector[0..4]
        .try_into()
        .expect("Slice with wrong length")
}

fn write_program(source: &str, writer: &mut impl Write) -> Result<(), Box<dyn std::error::Error>> {
    let mut compiler = Compiler::new();
    let stmts = statements_finish(Span::new(source)).map_err(|e| {
        format!(
            "{}:{}: {}",
            e.input.location_line(),
            e.input.get_utf8_column(),
            e
        )
    })?;

    println!("AST: {stmts:#?}");

    match type_check(&stmts, &mut TypeCheckContext::new()) {
        Ok(_) => println!("Typecheck Ok"),
        Err(e) => {
            return Err(format!(
                "{}:{}: {}",
                e.span.location_line(),
                e.span.get_utf8_column(),
                e
            )
            .into())
        }
    }

    compiler.compile(&stmts)?;

    // compiler.disasm(&mut std::io::stdout())?;

    // compiler.write_funcs(writer)?;

    Ok(())
}

fn compile(writer: &mut impl Write, source: String) -> Result<(), Box<dyn std::error::Error>> {
    write_program(&source, writer)
}

enum FnDef {
    User(FnByteCode),
    Native(NativeFn<'static>),
}

type Functions<'src> = HashMap<String, FnDecl<'src>>;

pub fn standard_functions<'src>() -> Functions<'src> {
    let funcs = Functions::new();

    funcs
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("no file name provided");
        std::process::exit(1);
    }

    let file_name = &args[1];

    let mut src_file = File::open(file_name)?;

    let mut source = String::new();

    src_file.read_to_string(&mut source)?;

    let mut buf = vec![];
    if let Err(e) = compile(&mut std::io::Cursor::new(&mut buf), source) {
        eprintln!("Compile Error: {e}");
        return Ok(());
    }
    // let bytecode = read_program(&mut std::io::Cursor::new(&mut buf))?;
    // if let Err(e) = bytecode.interpret("main", &[]) {
    //     eprintln!("Runtime error: {e:?}");
    // }
    Ok(())
}
