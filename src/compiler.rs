use std::{
    collections::HashMap,
    env,
    error::Error,
    fmt::Display,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
};

use primitive_types::U256;

use crate::parser::*;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum OpCode {
    LoadLiteral,
    Store,
    Copy,
    /// Duplicate the value on the top of the stack arg0 times
    Dup,
    Add,
    Sub,
    Mul,
    Div,
    Call,
    Jmp,
    /// Jump if false
    Jf,
    /// Pop a value from the stack, compare it with a value at arg0, push true if it's less
    Lt,
    /// Pop n values from the stack where n is given by arg0
    Pop,
    /// Return current function
    Ret,
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

impl_op_from!(
    LoadLiteral,
    Store,
    Copy,
    Dup,
    Add,
    Sub,
    Mul,
    Div,
    Call,
    Jmp,
    Jf,
    Lt,
    Pop,
    Ret
);

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Instruction {
    op: OpCode,
    arg0: u8,
}

impl Instruction {
    fn new(op: OpCode, arg0: u8) -> Self {
        Self { op, arg0 }
    }

    fn serialize(&self, writer: &mut impl Write) -> Result<(), std::io::Error> {
        writer.write_all(&[self.op as u8, self.arg0])?;
        Ok(())
    }

    fn deserialize(reader: &mut impl Read) -> Result<Self, std::io::Error> {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf)?;
        Ok(Self::new(buf[0].into(), buf[1]))
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

#[derive(Debug, Clone, Default)]
enum Target {
    #[default]
    Temp,
    Literal(usize),
    Local(String),
}

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

    fn write_insts(instructions: &[Instruction], writer: &mut impl Write) -> std::io::Result<()> {
        serialize_size(instructions.len(), writer)?;
        for instruction in instructions {
            instruction.serialize(writer).unwrap();
        }
        Ok(())
    }

    fn serialize(&self, writer: &mut impl Write) -> std::io::Result<()> {
        Self::write_args(&self.args, writer)?;
        Self::write_literals(&self.literals, writer)?;
        Self::write_insts(&self.instructions, writer)?;
        Ok(())
    }

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

    fn read_instructions(reader: &mut impl Read) -> std::io::Result<Vec<Instruction>> {
        let num_instructions = deserialize_size(reader)?;
        let mut instructions = Vec::with_capacity(num_instructions);
        for _ in 0..num_instructions {
            let inst = Instruction::deserialize(reader)?;
            instructions.push(inst);
        }
        Ok(instructions)
    }

    fn deserialize(reader: &mut impl Read) -> std::io::Result<Self> {
        let args = Self::read_args(reader)?;
        let literals = Self::read_literals(reader)?;
        let instructions = Self::read_instructions(reader)?;
        Ok(Self {
            args,
            literals,
            instructions,
        })
    }

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
    for (i, inst) in instructions.iter().enumerate() {
        match inst.op {
            LoadLiteral => writeln!(
                writer,
                "    [{i}] {:?} {} ({:?})",
                inst.op, inst.arg0, literals[inst.arg0 as usize]
            )?,
            Copy | Dup | Call | Jmp | Jf | Pop | Store | Ret => {
                writeln!(writer, "    [{i}] {:?} {}", inst.op, inst.arg0)?
            }
            _ => writeln!(writer, "    [{i}] {:?}", inst.op)?,
        }
    }
    Ok(())
}

struct Compiler {
    literals: Vec<Value>,
    instructions: Vec<Instruction>,
    target_stack: Vec<Target>,
    funcs: HashMap<String, FnByteCode>,
}

impl Compiler {
    fn new() -> Self {
        Self {
            literals: vec![],
            instructions: vec![],
            target_stack: vec![],
            funcs: HashMap::new(),
        }
    }

    fn stack_top(&self) -> StkIdx {
        StkIdx(self.target_stack.len() - 1)
    }

    fn add_literal(&mut self, value: Value) -> u8 {
        let existing = self
            .literals
            .iter()
            .enumerate()
            .find(|(_, val)| **val == value);
        if let Some((i, _)) = existing {
            i as u8
        } else {
            let ret = self.literals.len();
            self.literals.push(value);
            ret as u8
        }
    }

    /// Returns absolute position of inserted value
    fn add_inst(&mut self, op: OpCode, arg0: u8) -> InstPtr {
        let inst = self.instructions.len();
        self.instructions.push(Instruction { op, arg0 });
        InstPtr(inst)
    }

    fn add_copy_inst(&mut self, stack_idx: StkIdx) -> InstPtr {
        let inst = self.add_inst(
            OpCode::Copy,
            (self.target_stack.len() - stack_idx.0 - 1) as u8,
        );
        self.target_stack.push(Target::Temp);
        inst
    }

    fn add_load_literal_inst(&mut self, lit: u8) -> InstPtr {
        let inst = self.add_inst(OpCode::LoadLiteral, lit);
        self.target_stack.push(Target::Literal(lit as usize));
        inst
    }

    fn add_binop_inst(&mut self, op: OpCode) -> InstPtr {
        self.target_stack.pop();
        self.add_inst(op, 0)
    }

    fn add_store_inst(&mut self, stack_idx: StkIdx) -> InstPtr {
        if self.target_stack.len() < stack_idx.0 + 1 {
            eprintln!("Compiled bytecode so far:");
            disasm_common(&self.literals, &self.instructions, &mut std::io::stderr()).unwrap();
            panic!("Target stack undeflow during compilation!");
        }
        let inst = self.add_inst(
            OpCode::Store,
            (self.target_stack.len() - stack_idx.0 - 1) as u8,
        );
        self.target_stack.pop();
        inst
    }

    fn add_jf_inst(&mut self) -> InstPtr {
        // Push with jump address 0, because it will be set later
        let inst = self.add_inst(OpCode::Jf, 0);
        self.target_stack.pop();
        inst
    }

    fn fixup_jmp(&mut self, ip: InstPtr) {
        self.instructions[ip.0].arg0 = self.instructions.len() as u8;
    }

    /// Pop until given stack index
    fn add_pop_until_inst(&mut self, stack_idx: StkIdx) -> Option<InstPtr> {
        if self.target_stack.len() <= stack_idx.0 {
            return None;
        }
        let inst = self.add_inst(
            OpCode::Pop,
            (self.target_stack.len() - stack_idx.0 - 1) as u8,
        );
        self.target_stack.resize(stack_idx.0 + 1, Target::Temp);
        Some(inst)
    }

    fn add_fn(&mut self, name: String, args: &[(Span, TypeDecl)]) {
        self.funcs.insert(
            name,
            FnByteCode {
                args: args.iter().map(|(arg, _)| arg.to_string()).collect(),
                literals: std::mem::take(&mut self.literals),
                instructions: std::mem::take(&mut self.instructions),
            },
        );
    }

    fn write_funcs(&self, writer: &mut impl Write) -> std::io::Result<()> {
        serialize_size(self.funcs.len(), writer)?;
        for (name, func) in &self.funcs {
            serialize_str(name, writer)?;
            func.serialize(writer)?;
        }
        Ok(())
    }

    fn compile_expr(&mut self, ex: &Expression) -> Result<StkIdx, Box<dyn Error>> {
        Ok(match &ex.expr {
            ExprEnum::NumLiteral(num) => {
                let id = self.add_literal(Value::Uint256(*num));
                self.add_load_literal_inst(id);
                self.stack_top()
            }
            ExprEnum::StrLiteral(str) => {
                let id = self.add_literal(Value::Str(str.clone()));
                self.add_load_literal_inst(id);
                self.stack_top()
            }
            ExprEnum::BoolLiteral(b) => {
                let id = self.add_literal(Value::Bool(*b));
                self.add_load_literal_inst(id);
                self.stack_top()
            }
            ExprEnum::Ident(ident) => {
                let var = self.target_stack.iter().enumerate().find(|(_i, tgt)| {
                    if let Target::Local(id) = tgt {
                        id == ident.fragment()
                    } else {
                        false
                    }
                });
                if let Some(var) = var {
                    return Ok(StkIdx(var.0));
                } else {
                    return Err(format!("Variable not found: {ident:?}").into());
                }
            }
            ExprEnum::Add(lhs, rhs) => self.bin_op(OpCode::Add, lhs, rhs)?,
            ExprEnum::Sub(lhs, rhs) => self.bin_op(OpCode::Sub, lhs, rhs)?,
            ExprEnum::Mul(lhs, rhs) => self.bin_op(OpCode::Mul, lhs, rhs)?,
            ExprEnum::Div(lhs, rhs) => self.bin_op(OpCode::Div, lhs, rhs)?,
            ExprEnum::Gt(lhs, rhs) => self.bin_op(OpCode::Lt, rhs, lhs)?,
            ExprEnum::Lt(lhs, rhs) => self.bin_op(OpCode::Lt, lhs, rhs)?,
            ExprEnum::FnInvoke(name, args) => {
                let stack_before_args = self.target_stack.len();
                let name = self.add_literal(Value::Str(name.to_string()));
                let args = args
                    .iter()
                    .map(|arg| self.compile_expr(arg))
                    .collect::<Result<Vec<_>, _>>()?;

                let stack_before_call = self.target_stack.len();
                self.add_load_literal_inst(name);
                for arg in &args {
                    self.add_copy_inst(*arg);
                }

                self.add_inst(OpCode::Call, args.len() as u8);
                self.target_stack
                    .resize(stack_before_call + 1, Target::Temp);
                self.coerce_stack(StkIdx(stack_before_args));
                self.stack_top()
            }
            ExprEnum::If(cond, true_branch, false_branch) => {
                use OpCode::*;
                let cond = self.compile_expr(cond)?;
                self.add_copy_inst(cond);
                let jf_inst = self.add_jf_inst();
                let stack_size_before = self.target_stack.len();
                self.compile_stmts_or_zero(true_branch)?;
                self.coerce_stack(StkIdx(stack_size_before + 1));
                let jmp_inst = self.add_inst(Jmp, 0);
                self.fixup_jmp(jf_inst);
                self.target_stack.resize(stack_size_before, Target::Temp);
                if let Some(false_branch) = false_branch.as_ref() {
                    self.compile_stmts_or_zero(&false_branch)?;
                }
                self.coerce_stack(StkIdx(stack_size_before + 1));
                self.fixup_jmp(jmp_inst);
                self.stack_top()
            }
        })
    }

    fn bin_op(
        &mut self,
        op: OpCode,
        lhs: &Expression,
        rhs: &Expression,
    ) -> Result<StkIdx, Box<dyn Error>> {
        let lhs = self.compile_expr(lhs)?;
        let rhs = self.compile_expr(rhs)?;
        self.add_copy_inst(lhs);
        self.add_copy_inst(rhs);
        self.add_inst(op, 0);
        self.target_stack.pop();
        self.target_stack.pop();
        self.target_stack.push(Target::Temp);
        Ok(self.stack_top())
    }

    /// Coerce the stack size to be target + 1, and move the old top
    /// to the new top.
    fn coerce_stack(&mut self, target: StkIdx) {
        if target.0 < self.target_stack.len() - 1 {
            self.add_store_inst(target);
            self.add_pop_until_inst(target);
        } else if self.target_stack.len() - 1 < target.0 {
            for _ in self.target_stack.len() - 1..target.0 {
                self.add_copy_inst(self.stack_top());
            }
        }
    }

    fn compile_stmts(&mut self, stmts: &Statements) -> Result<Option<StkIdx>, Box<dyn Error>> {
        let mut last_result = None;
        for stmt in stmts {
            match stmt {
                Statement::Expression(ex) => {
                    last_result = Some(self.compile_expr(ex)?);
                }
                Statement::VarDef { name, ex, .. } => {
                    let mut ex = self.compile_expr(ex)?;
                    if !matches!(self.target_stack[ex.0], Target::Temp) {
                        self.add_copy_inst(ex);
                        ex = self.stack_top();
                    }
                    self.target_stack[ex.0] = Target::Local(name.to_string());
                }
                Statement::VarAssign { name, ex, .. } => {
                    let stk_ex = self.compile_expr(ex)?;
                    let (stk_local, _) = self
                        .target_stack
                        .iter_mut()
                        .enumerate()
                        .find(|(_, tgt)| {
                            if let Target::Local(tgt) = tgt {
                                tgt == name.fragment()
                            } else {
                                false
                            }
                        })
                        .ok_or_else(|| format!("Variable name not found: {name}"))?;
                    self.add_copy_inst(stk_ex);
                    self.add_store_inst(StkIdx(stk_local));
                }
                Statement::FnDef {
                    name, args, stmts, ..
                } => {
                    let literals = std::mem::take(&mut self.literals);
                    let instructions = std::mem::take(&mut self.instructions);
                    let target_stack = std::mem::take(&mut self.target_stack);
                    self.target_stack = args
                        .iter()
                        .map(|arg| Target::Local(arg.0.to_string()))
                        .collect();
                    self.compile_stmts(stmts)?;
                    self.add_fn(name.to_string(), args);
                    self.literals = literals;
                    self.instructions = instructions;
                    self.target_stack = target_stack;
                }
                Statement::Return(ex) => {
                    let res = self.compile_expr(ex)?;
                    self.add_inst(OpCode::Ret, (self.target_stack.len() - res.0 - 1) as u8);
                }
            }
        }
        Ok(last_result)
    }

    fn compile_stmts_or_zero(&mut self, stmts: &Statements) -> Result<StkIdx, Box<dyn Error>> {
        Ok(self.compile_stmts(stmts)?.unwrap_or_else(|| {
            let id = self.add_literal(Value::Uint256(U256::zero()));
            self.add_load_literal_inst(id);
            self.stack_top()
        }))
    }

    fn compile(&mut self, stmts: &Statements) -> Result<(), Box<dyn std::error::Error>> {
        let name = "main";
        self.compile_stmts_or_zero(stmts)?;
        self.add_fn(name.to_string(), &[]);
        Ok(())
    }

    fn disasm(&self, writer: &mut impl Write) -> std::io::Result<()> {
        for (name, fn_def) in &self.funcs {
            writeln!(writer, "Function {name:?}:")?;
            fn_def.disasm(writer)?;
        }
        Ok(())
    }
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

    compiler.disasm(&mut std::io::stdout())?;

    compiler.write_funcs(writer)?;

    Ok(())
}

struct ByteCode {
    funcs: HashMap<String, FnDef>,
}

impl ByteCode {
    fn new() -> Self {
        Self {
            funcs: HashMap::new(),
        }
    }

    fn read_funcs(&mut self, reader: &mut impl Read) -> std::io::Result<()> {
        let num_funcs = deserialize_size(reader)?;
        let mut funcs: HashMap<_, _> = standard_functions()
            .into_iter()
            .filter_map(|(name, f)| {
                if let FnDecl::Native(f) = f {
                    Some((name, FnDef::Native(f)))
                } else {
                    None
                }
            })
            .collect();
        for _ in 0..num_funcs {
            let name = deserialize_str(reader)?;
            funcs.insert(name, FnDef::User(FnByteCode::deserialize(reader)?));
        }
        self.funcs = funcs;
        Ok(())
    }

    fn interpret(
        &self,
        fn_name: &str,
        args: &[Value],
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let fn_def = self
            .funcs
            .get(fn_name)
            .ok_or_else(|| format!("Function {fn_name:?} was not found"))?;
        let fn_def = match fn_def {
            FnDef::User(user) => user,
            FnDef::Native(n) => return Ok((*n.code)(args)),
        };
        let mut stack = args.to_vec();
        let mut ip = 0;

        while ip < fn_def.instructions.len() {
            let instruction = &fn_def.instructions[ip];
            match instruction.op {
                OpCode::LoadLiteral => {
                    stack.push(fn_def.literals[instruction.arg0 as usize].clone());
                }
                OpCode::Store => {
                    let idx = stack.len() - instruction.arg0 as usize - 1;
                    let value = stack.pop().expect("Store needs an argument");
                    stack[idx] = value;
                }
                OpCode::Copy => {
                    stack.push(stack[stack.len() - instruction.arg0 as usize - 1].clone());
                }
                OpCode::Dup => {
                    let top = stack.last().unwrap().clone();
                    stack.extend((0..instruction.arg0).map(|_| top.clone()));
                }
                OpCode::Add => {}
                OpCode::Sub => {}
                OpCode::Mul => {}
                OpCode::Div => {}
                OpCode::Call => {
                    let args = &stack[stack.len() - instruction.arg0 as usize..];
                    let fname = &stack[stack.len() - instruction.arg0 as usize - 1];
                    let Value::Str(fname) = fname else {
                        panic!("Function name shall be a string: {fname:?}");
                    };
                    let res = self.interpret(fname, args)?;
                    stack.resize(
                        stack.len() - instruction.arg0 as usize - 1,
                        Value::Uint256(U256::zero()),
                    );
                    stack.push(res);
                }
                OpCode::Jmp => {
                    ip = instruction.arg0 as usize;
                    continue;
                }
                OpCode::Jf => {
                    let cond = stack.pop().expect("Jf needs an argument");
                }
                OpCode::Lt => {}
                OpCode::Pop => {
                    stack.resize(stack.len() - instruction.arg0 as usize, Value::default());
                }
                OpCode::Ret => {
                    return Ok(stack
                        .get(stack.len() - instruction.arg0 as usize - 1)
                        .ok_or_else(|| "Stack underflow".to_string())?
                        .clone());
                }
            }
            ip += 1;
        }

        Ok(stack.pop().ok_or_else(|| "Stack underflow".to_string())?)
    }

    fn interpret_bin_op_str(
        &self,
        stack: &mut Vec<Value>,
        op_uint256: impl FnOnce(U256, U256) -> U256,
        op_str: impl FnOnce(&str, &str) -> Option<String>,
    ) {
        use Value::*;
        let rhs = stack.pop().expect("Stack underflow");
        let lhs = stack.pop().expect("Stack underflow");
        let res = match (lhs, rhs) {
            (Uint256(lhs), Uint256(rhs)) => Uint256(op_uint256(lhs, rhs)),

            (Str(lhs), Str(rhs)) => {
                if let Some(res) = op_str(&lhs, &rhs) {
                    Str(res)
                } else {
                    panic!("Incompatible types in binary op: {lhs:?} and {rhs:?}");
                }
            }
            (lhs, rhs) => panic!("Incompatible types in binary op: {lhs:?} and {rhs:?}"),
        };
        stack.push(res);
    }
}

fn compile(writer: &mut impl Write, source: String) -> Result<(), Box<dyn std::error::Error>> {
    write_program(&source, writer)
}

fn read_program(reader: &mut impl Read) -> std::io::Result<ByteCode> {
    let mut bytecode = ByteCode::new();
    bytecode.read_funcs(reader)?;
    Ok(bytecode)
}

enum FnDef {
    User(FnByteCode),
    Native(NativeFn<'static>),
}

type Functions<'src> = HashMap<String, FnDecl<'src>>;

pub fn standard_functions<'src>() -> Functions<'src> {
    let mut funcs = Functions::new();

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
    let bytecode = read_program(&mut std::io::Cursor::new(&mut buf))?;
    if let Err(e) = bytecode.interpret("main", &[]) {
        eprintln!("Runtime error: {e:?}");
    }
    Ok(())
}
