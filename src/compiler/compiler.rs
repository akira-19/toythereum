use std::{
    collections::HashMap,
    env,
    error::Error,
    fs::{self, File},
    io::{Read, Write},
    path::Path,
    vec,
};

use primitive_types::U256;
use tiny_keccak::{Hasher, Keccak};

use crate::parser::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpCode {
    Stop = 0x00,
    Add = 0x01,
    Mul = 0x02,
    Sub = 0x03,
    Div = 0x04,

    Lt = 0x10,

    EQ = 0x14,
    SHR = 0x1C,

    CalldataLoad = 0x35,

    CodeCopy = 0x39,

    MLoad = 0x51,
    MStore = 0x52,
    SLoad = 0x54,
    SStore = 0x55,
    Jump = 0x56,
    JumpI = 0x57,
    JumpDest = 0x5B,

    Push0 = 0x5F,
    Push1 = 0x60,
    Push32 = 0x7F,

    Dup1 = 0x80,

    Return = 0xF3,
    Revert = 0xFD,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArgValue {
    U256(U256),
    FnSelector([u8; 4]),
    CodeLength,
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

#[derive(Debug, Clone)]
enum Valuable {
    Value(Value),
    Undetermined(TypeDecl),
}

#[derive(Debug, Clone)]
struct MemoryVariableTable {
    value: Valuable,
    address: U256,
}

#[derive(Debug, Clone)]
struct StorageVariableTable {
    value_type: TypeDecl,
    slot: U256,
}

#[derive(Debug, Clone)]
struct Arg {
    name: String,
    value_type: TypeDecl,
}

#[derive(Debug, Clone)]
struct Function<'a> {
    statements: &'a Vec<Statement<'a>>,
    args: Vec<Arg>,
    memories: HashMap<String, MemoryVariableTable>,
    memory_pointer: U256,
    selector: [u8; 4],
}

struct Compiler<'a> {
    literals: Vec<Value>,
    instructions: Vec<Instruction>,
    funcs: Vec<Function<'a>>,
    storages: HashMap<String, StorageVariableTable>,
    slot: U256,
    pc: usize,
}

impl<'a> Compiler<'a> {
    fn new() -> Self {
        Self {
            literals: vec![],
            instructions: vec![],
            funcs: vec![],
            storages: HashMap::new(),
            slot: U256::zero(),
            pc: 0,
        }
    }

    // Returns absolute position of inserted value
    fn add_inst(&mut self, op: OpCode, arg0: Option<ArgValue>) -> InstPtr {
        let inst = self.instructions.len();
        self.instructions.push(Instruction::new(op, arg0));
        self.pc += 1;
        if arg0.is_some() {
            self.pc += 32;
        }
        InstPtr(inst)
    }

    fn load_memory(&mut self, v: &MemoryVariableTable) -> Result<(), Box<dyn Error>> {
        // TODO: if the value is dynamic type, the memory should be loaded from multiple slots
        self.add_inst(OpCode::Push32, Some(ArgValue::U256(v.address)));
        self.add_inst(OpCode::MLoad, None);
        Ok(())
    }

    fn save_memory(
        &mut self,
        name: String,
        v: &Valuable,
        selector: [u8; 4],
    ) -> Result<(U256, U256), Box<dyn Error>> {
        // TODO: if the value is dynamic type, we need to user more than one memory slot
        // but for now, only one slot will be used

        let func = self.find_func(selector)?;
        let memory_pointer = func.memory_pointer.clone();

        func.memories.insert(
            name,
            MemoryVariableTable {
                value: v.clone(),
                address: memory_pointer,
            },
        );
        func.memory_pointer += U256::from(32);

        self.add_inst(OpCode::Push32, Some(ArgValue::U256(memory_pointer)));
        self.add_inst(OpCode::MStore, None);

        Ok((memory_pointer, U256::from(32)))
    }

    fn update_memory(
        &mut self,
        name: String,
        v: &Valuable,
        selector: [u8; 4],
    ) -> Result<(), Box<dyn Error>> {
        let func = self.find_func(selector)?;
        if let Some(m) = func.memories.get_mut(&name) {
            let address = m.address;
            m.value = v.clone();

            // TODO: if the value is dynamic type, new memory slots might be needed
            self.add_inst(OpCode::Push32, Some(ArgValue::U256(address)));
            self.add_inst(OpCode::MStore, None);
            Ok(())
        } else {
            Err(format!("Variable not found: {name:?}").into())
        }
    }

    fn load_storage(&mut self, v: &StorageVariableTable) -> Result<(), Box<dyn Error>> {
        self.add_inst(OpCode::Push32, Some(ArgValue::U256(v.slot)));
        match &v.value_type {
            TypeDecl::Uint256 => {
                self.add_inst(OpCode::SLoad, None);
            }
            TypeDecl::Str => {
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
            TypeDecl::Bool => {
                self.add_inst(OpCode::SLoad, None);
            }
        }

        Ok(())
    }

    fn save_storage(&mut self, name: String, v: &Valuable) -> Result<(), Box<dyn Error>> {
        let value_type;
        let slot = self.slot.clone();
        match v {
            Valuable::Value(value) => {
                match value {
                    Value::Uint256(value) => {
                        self.add_inst(OpCode::Push32, Some(ArgValue::U256(*value)));
                        self.add_inst(OpCode::Push32, Some(ArgValue::U256(self.slot)));
                        self.add_inst(OpCode::SStore, None);
                        self.slot += U256::one();
                        value_type = TypeDecl::Uint256;
                    }
                    Value::Str(value) => {
                        // store the length of the string
                        let len = value.clone().into_bytes().len();
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
                        value_type = TypeDecl::Str;
                    }
                    Value::Bool(value) => {
                        let num = if *value { U256::one() } else { U256::zero() };
                        self.add_inst(OpCode::Push32, Some(ArgValue::U256(num)));
                        self.add_inst(OpCode::Push32, Some(ArgValue::U256(self.slot)));
                        self.add_inst(OpCode::SStore, None);
                        self.slot += U256::one();
                        value_type = TypeDecl::Bool;
                    }
                }
            }
            Valuable::Undetermined(type_decl) => {
                value_type = type_decl.clone();
                match type_decl {
                    TypeDecl::Uint256 => {
                        self.add_inst(OpCode::Push32, Some(ArgValue::U256(self.slot)));
                        self.add_inst(OpCode::SStore, None);
                        self.slot += U256::one();
                    }
                    TypeDecl::Str => {
                        // store the length of the string
                        // data is already stored in the stack
                        // top of the stack is the length of the string
                        self.add_inst(OpCode::Push32, Some(ArgValue::U256(self.slot)));
                        self.add_inst(OpCode::SStore, None);
                        self.slot += U256::one();

                        // store the string
                        // TODO: implement some process if the string is longer than 32 bytes
                        // but for now, only one slot will be used

                        let mut hasher = Keccak::v256();
                        let mut pointer = [0u8; 32];
                        hasher.update(&self.slot.to_big_endian().to_vec());
                        hasher.finalize(&mut pointer);

                        // data is already stored on the top of the stack
                        // if it is longer than 32 bytes, it should be stored in multiple slots
                        self.add_inst(
                            OpCode::Push32,
                            Some(ArgValue::U256(U256::from_big_endian(&pointer))),
                        );
                        self.add_inst(OpCode::SStore, None);
                    }
                    TypeDecl::Bool => {
                        self.add_inst(OpCode::Push32, Some(ArgValue::U256(self.slot)));
                        self.add_inst(OpCode::SStore, None);
                        self.slot += U256::one();
                    }
                }
            }
        }

        self.storages
            .insert(name, StorageVariableTable { value_type, slot });

        Ok(())
    }

    fn update_storage(&mut self, name: String) -> Result<(), Box<dyn Error>> {
        if let Some(s) = self.storages.get(&name) {
            let slot = s.slot;
            match s.value_type {
                TypeDecl::Uint256 => {
                    self.add_inst(OpCode::Push32, Some(ArgValue::U256(slot)));
                    self.add_inst(OpCode::SStore, None);
                    self.slot += U256::one();
                }
                TypeDecl::Str => {
                    // store the length of the string
                    // data is already stored in the stack
                    // top of the stack is the length of the string
                    self.add_inst(OpCode::Push32, Some(ArgValue::U256(slot)));
                    self.add_inst(OpCode::SStore, None);
                    self.slot += U256::one();

                    // store the string
                    // TODO: implement some process if the string is longer than 32 bytes
                    // but for now, only one slot will be used

                    let mut hasher = Keccak::v256();
                    let mut pointer = [0u8; 32];
                    hasher.update(&self.slot.to_big_endian().to_vec());
                    hasher.finalize(&mut pointer);

                    // data is already stored on the top of the stack
                    // if it is longer than 32 bytes, it should be stored in multiple slots
                    self.add_inst(
                        OpCode::Push32,
                        Some(ArgValue::U256(U256::from_big_endian(&pointer))),
                    );
                    self.add_inst(OpCode::SStore, None);
                }
                TypeDecl::Bool => {
                    self.add_inst(OpCode::Push32, Some(ArgValue::U256(slot)));
                    self.add_inst(OpCode::SStore, None);
                    self.slot += U256::one();
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
        selector: Option<[u8; 4]>,
    ) -> Result<Valuable, Box<dyn Error>> {
        let r_value = self.compile_expr(rhs, selector)?;
        let l_value = self.compile_expr(lhs, selector)?;

        match (l_value, r_value) {
            (Valuable::Value(l), Valuable::Value(r)) => {
                let l = match l {
                    Value::Uint256(v) => v,
                    _ => return Err("Invalid type l".into()),
                };
                let r = match r {
                    Value::Uint256(v) => v,
                    _ => return Err("Invalid type r".into()),
                };
                let result = match op {
                    OpCode::Add => Value::Uint256(l + r),
                    OpCode::Sub => Value::Uint256(l - r),
                    OpCode::Mul => Value::Uint256(l * r),
                    OpCode::Div => {
                        if r == U256::zero() {
                            return Err("Division by zero".into());
                        }
                        Value::Uint256(l / r)
                    }
                    OpCode::Lt => {
                        if l < r {
                            Value::Bool(true)
                        } else {
                            Value::Bool(false)
                        }
                    }
                    _ => return Err("Invalid opcode".into()),
                };
                Ok(Valuable::Value(result))
            }
            (_, Valuable::Undetermined(TypeDecl::Uint256))
            | (Valuable::Undetermined(TypeDecl::Uint256), _) => {
                self.add_inst(op, None);
                Ok(Valuable::Undetermined(TypeDecl::Uint256))
            }
            _ => Err("Invalid type".into()),
        }
    }

    fn compile_expr(
        &mut self,
        ex: &Expression,
        selector: Option<[u8; 4]>,
    ) -> Result<Valuable, Box<dyn Error>> {
        let func = if let Some(value) = selector {
            Some(self.find_func(value)?.clone())
        } else {
            None
        };
        Ok(match &ex.expr {
            ExprEnum::NumLiteral(num) => {
                self.add_inst(OpCode::Push32, Some(ArgValue::U256(U256::from(*num))));
                return Ok(Valuable::Value(Value::Uint256(U256::from(*num))));
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
                if let Some(f) = func {
                    let arg = f
                        .args
                        .iter()
                        .enumerate()
                        .find(|(_, arg)| arg.name == ident.fragment().to_string());

                    if let Some((i, v)) = arg {
                        let offset = 32 * i + 4;
                        self.add_inst(OpCode::Push32, Some(ArgValue::U256(U256::from(offset))));
                        self.add_inst(OpCode::CalldataLoad, None);
                        return Ok(Valuable::Undetermined(v.value_type.clone()));
                    } else {
                        let v = f.memories.get(&ident.fragment().to_string()).map(|v| {
                            self.load_memory(v);
                            v.value.clone()
                        });
                        if let Some(v) = v {
                            return Ok(v);
                        }
                    }
                }

                if let Some(v) = self.storages.get(&ident.fragment().to_string()).cloned() {
                    self.load_storage(&v)?;
                    return Ok(Valuable::Undetermined(v.value_type));
                } else {
                    return Err(format!("Variable not found: {ident:?}").into());
                };
            }
            ExprEnum::Add(lhs, rhs) => self.bin_op(OpCode::Add, lhs, rhs, selector)?,
            ExprEnum::Sub(lhs, rhs) => self.bin_op(OpCode::Sub, lhs, rhs, selector)?,
            ExprEnum::Mul(lhs, rhs) => self.bin_op(OpCode::Mul, lhs, rhs, selector)?,
            ExprEnum::Div(lhs, rhs) => self.bin_op(OpCode::Div, lhs, rhs, selector)?,
            ExprEnum::Gt(lhs, rhs) => self.bin_op(OpCode::Lt, rhs, lhs, selector)?,
            ExprEnum::Lt(lhs, rhs) => self.bin_op(OpCode::Lt, lhs, rhs, selector)?,

            _ => todo!(),
        })
    }

    fn compile_stmts(
        &mut self,
        stmts: &'a Statements<'a>,
        selector: Option<[u8; 4]>,
    ) -> Result<Option<StkIdx>, Box<dyn Error>> {
        for stmt in stmts {
            match stmt {
                Statement::Expression(ex) => {
                    Some(self.compile_expr(ex, selector)?);
                }
                Statement::VarDef { name, ex, .. } => {
                    let result = self.compile_expr(ex, selector)?;
                    let var_name = name.fragment().to_string();

                    if let Some(selector) = selector {
                        self.save_memory(var_name, &result, selector)?;
                    } else {
                        self.save_storage(var_name, &result)?;
                    }
                }
                Statement::VarAssign { name, ex, .. } => {
                    let result = self.compile_expr(ex, selector)?;
                    let var_name = name.fragment().to_string();
                    if let Some(selector) = selector {
                        self.update_memory(var_name.clone(), &result, selector)?;
                    }

                    if let Some(_) = self.storages.get(&var_name) {
                        self.update_storage(var_name)?;
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

                    // save the func temporarily
                    let fn_args = args
                        .iter()
                        .map(|arg| Arg {
                            name: arg.0.fragment().to_string(),
                            value_type: arg.1,
                        })
                        .collect::<Vec<_>>();

                    let func = Function {
                        statements: stmts,
                        args: fn_args,
                        memories: HashMap::new(),
                        memory_pointer: U256::from_str_radix("80", 16)?,
                        selector,
                    };

                    self.funcs.push(func);
                }
                Statement::Return(ex) => {
                    let res = self.compile_expr(ex, selector)?;
                    if let Some(selector) = selector {
                        let (pointer, length) =
                            self.save_memory("return".to_string(), &res, selector)?;
                        self.add_inst(OpCode::Push32, Some(ArgValue::U256(length)));
                        self.add_inst(OpCode::Push32, Some(ArgValue::U256(pointer)));
                        self.add_inst(OpCode::Return, None);
                    } else {
                        return Err(format!("Variable not found.").into());
                    }
                }
            }
        }

        Ok(None)
        // Ok(last_result)
    }

    fn compile_func(&mut self, selector: [u8; 4]) -> Result<(), Box<dyn std::error::Error>> {
        // allocate the current pc to jump destination
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

        let func = self.find_func(selector)?.clone();
        self.compile_stmts(func.statements, Some(selector))?;
        Ok(())
    }

    fn compile(&mut self, stmts: &'a Statements<'a>) -> Result<(), Box<dyn std::error::Error>> {
        self.init_memory()?;
        self.compile_stmts(stmts, None)?;

        // add code deploy instructions
        // [16] PUSH2 0x0143
        // [17] DUP1
        // [18] PUSH2 0x0020
        // [19] PUSH0 0x
        // [20] CODECOPY
        // [21] PUSH0 0x
        // [22] RETURN
        self.add_inst(OpCode::Push32, Some(ArgValue::CodeLength));
        self.add_inst(OpCode::Dup1, None);

        let mut offset: u32 = self.calc_code_length();

        offset += 37; // 37 is the size of the instructions from push offset to return

        self.add_inst(OpCode::Push32, Some(ArgValue::U256(U256::from(offset))));
        self.add_inst(OpCode::Push0, None);
        self.add_inst(OpCode::CodeCopy, None);
        self.add_inst(OpCode::Push0, None);
        self.add_inst(OpCode::Return, None);

        // *****************************************************
        // following code is for the deployed contract code
        // *****************************************************
        self.init_memory()?;

        for func in self.funcs.clone() {
            self.add_inst(
                OpCode::Push32,
                Some(ArgValue::U256(U256::from_str_radix("E0", 16).unwrap())),
            );
            // get func selector from calldata
            self.add_inst(OpCode::Push32, Some(ArgValue::U256(U256::zero())));
            self.add_inst(OpCode::CalldataLoad, None);
            self.add_inst(OpCode::SHR, None);

            // add this function's selector to the stack and compare
            self.add_inst(
                OpCode::Push32,
                Some(ArgValue::U256(U256::from_big_endian(&func.selector))),
            );
            self.add_inst(OpCode::EQ, None);

            // if the selector is equal, jmp
            self.add_inst(
                OpCode::Push32,
                Some(ArgValue::FnSelector(func.selector.clone())),
            );
            self.add_inst(OpCode::JumpI, None);
        }

        // no function matched
        // not loading the memory for now
        self.add_inst(OpCode::Push32, Some(ArgValue::U256(U256::from(0))));
        self.add_inst(OpCode::Push32, Some(ArgValue::U256(U256::from(0))));
        self.add_inst(OpCode::Revert, None);

        // compile the functions
        for func in self.funcs.clone() {
            self.compile_func(func.selector)?;
        }

        let code_length = self.calc_code_length() - offset;

        self.instructions.iter_mut().for_each(|inst| {
            if inst.arg0 == Some(ArgValue::CodeLength) {
                inst.arg0 = Some(ArgValue::U256(U256::from(code_length)));
            }
        });

        for inst in &self.instructions {
            println!("{:?}", inst);
        }
        Ok(())
    }

    fn find_func(
        &mut self,
        selector: [u8; 4],
    ) -> Result<&mut Function<'a>, Box<dyn std::error::Error>> {
        if let Some(func) = self.funcs.iter_mut().find(|f| f.selector == selector) {
            Ok(func)
        } else {
            Err("Function not found".into())
        }
    }

    fn calc_code_length(&self) -> u32 {
        self.instructions
            .clone()
            .iter()
            .map(|inst| {
                let mut size = 1;
                if let Some(ArgValue::U256(_)) = inst.arg0 {
                    size += 32;
                } else if let Some(ArgValue::CodeLength) = inst.arg0 {
                    size += 32;
                } else if let Some(ArgValue::FnSelector(_)) = inst.arg0 {
                    size += 32;
                }
                size
            })
            .sum()
    }

    fn init_memory(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let memory_pointer = U256::from_str_radix("80", 16)?;
        self.add_inst(OpCode::Push32, Some(ArgValue::U256(memory_pointer)));
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

    compiler.instructions.iter().for_each(|inst| {
        writer.write_all(&[inst.op as u8]).unwrap();
        if let Some(arg) = inst.arg0 {
            match arg {
                ArgValue::U256(v) => {
                    writer.write_all(&v.to_big_endian().to_vec()).unwrap();
                }
                _ => {}
            }
        }
    });

    Ok(())
}

fn compile(writer: &mut impl Write, source: String) -> Result<(), Box<dyn std::error::Error>> {
    write_program(&source, writer)
}

type Functions<'src> = HashMap<String, FnDecl<'src>>;

pub fn standard_functions<'src>() -> Functions<'src> {
    let funcs = Functions::new();

    funcs
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    println!("{:?}", args);

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
    let hex_string: String = buf.iter().map(|b| format!("{:02x}", b)).collect();

    fs::create_dir_all("bytecode")?;

    let path_without_extension = file_name.strip_suffix(".sol").unwrap();
    let position = path_without_extension.rfind('/').unwrap();
    let file = &path_without_extension[(position + 1)..];
    let path = format!("bytecode/{}", file);

    fs::write(path, hex_string)?;

    Ok(())
}
