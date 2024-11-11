use std::{
    collections::HashMap,
    error::Error,
    fmt::Display,
    io::{Read, Write},
};

use nom_locate::LocatedSpan;

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, digit1, multispace0, multispace1, none_of},
    combinator::{cut, map_res, opt, recognize},
    error::ParseError,
    multi::{fold_many0, many0, separated_list0},
    sequence::{delimited, pair, preceded, terminated},
    Finish, IResult, InputTake, Offset, Parser,
};
use primitive_types::U256;

use crate::compiler::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Uint256(U256),
    Str(String),
    Bool(bool),
}

impl Default for Value {
    fn default() -> Self {
        Self::Uint256(U256::zero())
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uint256(value) => write!(f, "{value}"),
            Self::Str(value) => write!(f, "{value}"),
            Self::Bool(value) => write!(f, "{value}"),
        }
    }
}

impl Value {
    pub fn kind(&self) -> ValueKind {
        match self {
            Self::Uint256(_) => ValueKind::Uint256,
            Self::Str(_) => ValueKind::Str,
            Self::Bool(_) => ValueKind::Bool,
        }
    }

    pub fn to_vec_u8(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.serialize(&mut buf).unwrap();
        buf
    }

    pub fn serialize(&self, writer: &mut impl Write) -> std::io::Result<()> {
        let kind = self.kind() as u8;
        writer.write_all(&[kind])?;
        match self {
            Self::Uint256(value) => {
                writer.write_all(&value.to_big_endian())?;
            }
            Self::Str(value) => {
                serialize_str(value, writer)?;
            }
            Self::Bool(value) => {
                writer.write_all(&[if *value { 1 } else { 0 }])?;
            }
        }
        Ok(())
    }

    #[allow(non_upper_case_globals)]
    pub fn deserialize(reader: &mut impl Read) -> std::io::Result<Self> {
        const Uint256: u8 = ValueKind::Uint256 as u8;
        const Str: u8 = ValueKind::Str as u8;
        const Bool: u8 = ValueKind::Bool as u8;

        let mut kind_buf = [0u8; 1];
        reader.read_exact(&mut kind_buf)?;
        match kind_buf[0] {
            Uint256 => {
                let mut buf = [0u8; 32];
                reader.read_exact(&mut buf)?;
                Ok(Value::Uint256(U256::from_big_endian(&buf)))
            }
            Str => Ok(Value::Str(deserialize_str(reader)?)),
            Bool => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(Value::Bool(buf[0] != 0))
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "ValueKind {} does not match to any known value",
                    kind_buf[0]
                ),
            )),
        }
    }
}

#[repr(u8)]
pub enum ValueKind {
    Uint256,
    Str,
    Bool,
}

pub type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TypeDecl {
    Uint256,
    Str,
    Bool,
}

impl TypeDecl {
    pub fn type_name(&self) -> &str {
        use TypeDecl::*;
        match self {
            Uint256 => "uint256",
            Str => "string",
            Bool => "bool",
        }
    }
}

fn tc_check_type<'src>(
    value: &Option<TypeDecl>,
    target: &Option<TypeDecl>,
    span: Span<'src>,
) -> Result<Option<TypeDecl>, TypeCheckError<'src>> {
    if value == target {
        Ok(*value)
    } else {
        Err(TypeCheckError::new(
            format!("{:?} cannot be assigned to {:?}", value, target),
            span,
        ))
    }
}

pub struct TypeCheckContext<'src, 'ctx> {
    /// Variables table for type checking.
    vars: HashMap<&'src str, TypeDecl>,
    /// Function names are owned strings because it can be either from source or native.
    funcs: HashMap<String, FnDecl<'src>>,
    super_context: Option<&'ctx TypeCheckContext<'src, 'ctx>>,
}

impl<'src, 'ctx> TypeCheckContext<'src, 'ctx> {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            funcs: standard_functions(),
            super_context: None,
        }
    }

    pub fn get_var(&self, name: &str) -> Option<TypeDecl> {
        if let Some(val) = self.vars.get(name) {
            Some(val.clone())
        } else if let Some(super_ctx) = self.super_context {
            super_ctx.get_var(name)
        } else {
            None
        }
    }

    pub fn get_fn(&self, name: &str) -> Option<&FnDecl<'src>> {
        if let Some(val) = self.funcs.get(name) {
            Some(val)
        } else if let Some(super_ctx) = self.super_context {
            super_ctx.get_fn(name)
        } else {
            None
        }
    }

    pub fn push_stack(super_ctx: &'ctx Self) -> Self {
        Self {
            vars: HashMap::new(),
            funcs: HashMap::new(),
            super_context: Some(super_ctx),
        }
    }
}

#[derive(Debug)]
pub struct TypeCheckError<'src> {
    pub msg: String,
    pub span: Span<'src>,
}

impl<'src> std::fmt::Display for TypeCheckError<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}\nlocation: {}:{}: {}",
            self.msg,
            self.span.location_line(),
            self.span.get_utf8_column(),
            self.span.fragment()
        )
    }
}

impl<'src> Error for TypeCheckError<'src> {}

impl<'src> TypeCheckError<'src> {
    fn new(msg: String, span: Span<'src>) -> Self {
        Self { msg, span }
    }
}

pub fn tc_binary_op<'src>(
    lhs: &Expression<'src>,
    rhs: &Expression<'src>,
    ctx: &mut TypeCheckContext<'src, '_>,
    op: &str,
) -> Result<TypeDecl, TypeCheckError<'src>> {
    let lhst = tc_expr(lhs, ctx)?;
    let rhst = tc_expr(rhs, ctx)?;
    binary_op_type(&lhst, &rhst).map_err(|_| {
        TypeCheckError::new(
            format!(
                "Operation {op} between incompatible type: {:?} and {:?}",
                lhst, rhst,
            ),
            lhs.span,
        )
    })
}

fn binary_op_type(lhs: &Option<TypeDecl>, rhs: &Option<TypeDecl>) -> Result<TypeDecl, ()> {
    use TypeDecl::*;
    Ok(match (lhs, rhs) {
        (Some(Uint256), Some(Uint256)) => Uint256,
        _ => return Err(()),
    })
}

fn tc_binary_cmp<'src>(
    lhs: &Expression<'src>,
    rhs: &Expression<'src>,
    ctx: &mut TypeCheckContext<'src, '_>,
    op: &str,
) -> Result<TypeDecl, TypeCheckError<'src>> {
    use TypeDecl::*;
    let lhst = tc_expr(lhs, ctx)?;
    let rhst = tc_expr(rhs, ctx)?;
    Ok(match (&lhst, &rhst) {
        (Some(Uint256), Some(Uint256)) => Bool,
        _ => {
            return Err(TypeCheckError::new(
                format!(
                    "Operation {op} between incompatible type: {:?} and {:?}",
                    lhst, rhst,
                ),
                lhs.span,
            ))
        }
    })
}

fn tc_expr<'src>(
    e: &Expression<'src>,
    ctx: &mut TypeCheckContext<'src, '_>,
) -> Result<Option<TypeDecl>, TypeCheckError<'src>> {
    use ExprEnum::*;
    Ok(match &e.expr {
        NumLiteral(_val) => Some(TypeDecl::Uint256),
        StrLiteral(_val) => Some(TypeDecl::Str),
        BoolLiteral(_val) => Some(TypeDecl::Bool),
        Ident(str) => {
            let v = ctx.get_var(str);
            if v == None {
                Err(TypeCheckError::new(
                    format!("Variable \"{}\" not found in scope", str),
                    e.span,
                ))?;
            }
            v
        }
        FnInvoke(str, args) => {
            let args_ty = args
                .iter()
                .map(|v| Ok((tc_expr(v, ctx)?, v.span)))
                .collect::<Result<Vec<_>, _>>()?;
            let func = ctx.get_fn(**str).ok_or_else(|| {
                TypeCheckError::new(format!("function {} is not defined", str), *str)
            })?;
            let args_decl = func.args();
            for ((arg_ty, arg_span), decl) in args_ty.iter().zip(args_decl.iter()) {
                tc_check_type(&arg_ty, &Some(decl.1), *arg_span)?;
            }
            func.ret_type()
        }
        Add(lhs, rhs) => Some(tc_binary_op(&lhs, &rhs, ctx, "Add")?),
        Sub(lhs, rhs) => Some(tc_binary_op(&lhs, &rhs, ctx, "Sub")?),
        Mul(lhs, rhs) => Some(tc_binary_op(&lhs, &rhs, ctx, "Mult")?),
        Div(lhs, rhs) => Some(tc_binary_op(&lhs, &rhs, ctx, "Div")?),
        Lt(lhs, rhs) => Some(tc_binary_cmp(&lhs, &rhs, ctx, "LT")?),
        Gt(lhs, rhs) => Some(tc_binary_cmp(&lhs, &rhs, ctx, "GT")?),
        If(cond, true_branch, false_branch) => {
            let t = tc_check_type(&tc_expr(cond, ctx)?, &Some(TypeDecl::Bool), cond.span)?;
            // todo: may be nothing to return
            t
        }
    })
}

pub fn type_check<'src>(
    stmts: &Vec<Statement<'src>>,
    ctx: &mut TypeCheckContext<'src, '_>,
) -> Result<Option<TypeDecl>, TypeCheckError<'src>> {
    for stmt in stmts {
        match stmt {
            Statement::VarDef { name, td, ex, .. } => {
                let init_type = tc_expr(ex, ctx)?;
                let init_type = tc_check_type(&init_type, &Some(*td), ex.span)?;

                if let Some(var_type) = init_type {
                    ctx.vars.insert(**name, var_type);
                } else {
                    return Err(TypeCheckError::new("Must have type".to_string(), ex.span));
                }
            }
            Statement::VarAssign { name, ex, .. } => {
                let init_type = tc_expr(ex, ctx)?;
                let target = ctx.vars.get(**name);
                if let Some(target) = target {
                    tc_check_type(&init_type, &Some(*target), ex.span)?;
                } else {
                    let target = ctx
                        .super_context
                        .unwrap()
                        .vars
                        .get(**name)
                        .expect("Variable not found");
                    tc_check_type(&init_type, &Some(*target), ex.span)?;
                    // return Err(TypeCheckError::new(
                    //     format!("Variable \"{}\" not found in scope", name),
                    //     ex.span,
                    // ));
                }
                // let target = ctx.vars.get(**name).expect("Variable not found");
                // tc_check_type(&init_type, &Some(*target), ex.span)?;
            }
            Statement::FnDef {
                name,
                args,
                ret_type,
                stmts,
            } => {
                // Function declaration needs to be added first to allow recursive calls
                ctx.funcs.insert(
                    name.to_string(),
                    FnDecl::User(UserFn {
                        args: args.clone(),
                        ret_type: *ret_type,
                    }),
                );
                let mut subctx = TypeCheckContext::push_stack(ctx);
                for (arg, ty) in args.iter() {
                    subctx.vars.insert(arg, *ty);
                }
                let last_stmt = type_check(stmts, &mut subctx)?;
                tc_check_type(&last_stmt, ret_type, stmts.span())?;
            }
            Statement::Expression(e) => {
                tc_expr(&e, ctx)?;
            }
            Statement::Return(e) => {
                return tc_expr(e, ctx);
            }
        }
    }
    Ok(None)
}

pub enum FnDecl<'src> {
    User(UserFn<'src>),
    Native(NativeFn<'src>),
}

impl<'src> FnDecl<'src> {
    pub fn args(&self) -> Vec<(&'src str, TypeDecl)> {
        match self {
            Self::User(user) => user
                .args
                .iter()
                .map(|(name, ty)| (*name.fragment(), *ty))
                .collect(),
            Self::Native(code) => code.args.clone(),
        }
    }

    pub fn ret_type(&self) -> Option<TypeDecl> {
        match self {
            Self::User(user) => user.ret_type,
            Self::Native(native) => native.ret_type,
        }
    }
}

pub struct UserFn<'src> {
    pub args: Vec<(Span<'src>, TypeDecl)>,
    pub ret_type: Option<TypeDecl>,
}

pub struct NativeFn<'src> {
    pub args: Vec<(&'src str, TypeDecl)>,
    pub ret_type: Option<TypeDecl>,
    pub code: Box<dyn Fn(&[Value]) -> Value>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum ExprEnum<'src> {
    Ident(Span<'src>),
    NumLiteral(U256),
    StrLiteral(String),
    BoolLiteral(bool),
    FnInvoke(Span<'src>, Vec<Expression<'src>>),
    Add(Box<Expression<'src>>, Box<Expression<'src>>),
    Sub(Box<Expression<'src>>, Box<Expression<'src>>),
    Mul(Box<Expression<'src>>, Box<Expression<'src>>),
    Div(Box<Expression<'src>>, Box<Expression<'src>>),
    Gt(Box<Expression<'src>>, Box<Expression<'src>>),
    Lt(Box<Expression<'src>>, Box<Expression<'src>>),
    If(
        Box<Expression<'src>>,
        Box<Statements<'src>>,
        Option<Box<Statements<'src>>>,
    ),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Expression<'a> {
    pub(crate) expr: ExprEnum<'a>,
    pub(crate) span: Span<'a>,
}

impl<'a> Expression<'a> {
    pub fn new(expr: ExprEnum<'a>, span: Span<'a>) -> Self {
        Self { expr, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement<'src> {
    Expression(Expression<'src>),
    VarDef {
        span: Span<'src>,
        name: Span<'src>,
        td: TypeDecl,
        ex: Expression<'src>,
        storage: bool,
    },
    VarAssign {
        span: Span<'src>,
        name: Span<'src>,
        ex: Expression<'src>,
    },
    FnDef {
        name: Span<'src>,
        args: Vec<(Span<'src>, TypeDecl)>,
        ret_type: Option<TypeDecl>,
        stmts: Statements<'src>,
    },
    Return(Expression<'src>),
}

impl<'src> Statement<'src> {
    pub fn span(&self) -> Option<Span<'src>> {
        use Statement::*;
        Some(match self {
            Expression(ex) => ex.span,
            VarDef { span, .. } => *span,
            VarAssign { span, .. } => *span,
            FnDef { name, stmts, .. } => calc_offset(*name, stmts.span()),
            Return(ex) => ex.span,
        })
    }
}

trait GetSpan<'a> {
    fn span(&self) -> Span<'a>;
}

pub type Statements<'a> = Vec<Statement<'a>>;

impl<'a> GetSpan<'a> for Statements<'a> {
    fn span(&self) -> Span<'a> {
        self.iter().find_map(|stmt| stmt.span()).unwrap()
    }
}

fn space_delimited<'src, O, E>(
    f: impl Parser<Span<'src>, O, E>,
) -> impl FnMut(Span<'src>) -> IResult<Span<'src>, O, E>
where
    E: ParseError<Span<'src>>,
{
    delimited(multispace0, f, multispace0)
}

/// Calculate offset between the start positions of the input spans and return a span between them.
///
/// Note: `i` shall start earlier than `r`, otherwise wrapping would occur.
fn calc_offset<'a>(i: Span<'a>, r: Span<'a>) -> Span<'a> {
    i.take(i.offset(&r))
}

fn factor(i: Span) -> IResult<Span, Expression> {
    alt((
        str_literal,
        num_literal,
        bool_literal,
        func_call,
        ident,
        parens,
    ))(i)
}

fn func_call(i: Span) -> IResult<Span, Expression> {
    let (r, ident) = space_delimited(identifier)(i)?;
    let (r, args) = space_delimited(delimited(
        tag("("),
        many0(delimited(multispace0, expr, space_delimited(opt(tag(","))))),
        tag(")"),
    ))(r)?;
    Ok((
        r,
        Expression {
            expr: ExprEnum::FnInvoke(ident, args),
            span: i,
        },
    ))
}

fn ident(input: Span) -> IResult<Span, Expression> {
    let (r, res) = space_delimited(identifier)(input)?;
    Ok((
        r,
        Expression {
            expr: ExprEnum::Ident(res),
            span: input,
        },
    ))
}

fn identifier(input: Span) -> IResult<Span, Span> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(input)
}

fn str_literal(i: Span) -> IResult<Span, Expression> {
    let (r0, _) = preceded(multispace0, char('\"'))(i)?;
    let (r, val) = many0(none_of("\""))(r0)?;
    let (r, _) = terminated(char('"'), multispace0)(r)?;
    Ok((
        r,
        Expression::new(
            ExprEnum::StrLiteral(
                val.iter()
                    .collect::<String>()
                    .replace("\\\\", "\\")
                    .replace("\\n", "\n"),
            ),
            i,
        ),
    ))
}

fn num_literal(input: Span) -> IResult<Span, Expression> {
    let (r, v) = space_delimited(digit1)(input)?;
    Ok((
        r,
        Expression::new(
            ExprEnum::NumLiteral(v.parse().map_err(|_| {
                nom::Err::Error(nom::error::Error {
                    input,
                    code: nom::error::ErrorKind::Digit,
                })
            })?),
            v,
        ),
    ))
}

fn bool_literal(input: Span) -> IResult<Span, Expression> {
    let (r, v) = alt((tag("true"), tag("false")))(input)?;
    Ok((
        r,
        Expression::new(
            ExprEnum::BoolLiteral(v.parse().map_err(|_| {
                nom::Err::Error(nom::error::Error {
                    input,
                    code: nom::error::ErrorKind::Tag,
                })
            })?),
            v,
        ),
    ))
}

fn parens(i: Span) -> IResult<Span, Expression> {
    space_delimited(delimited(tag("("), expr, tag(")")))(i)
}

fn term(i: Span) -> IResult<Span, Expression> {
    let (r, init) = factor(i)?;

    let res = fold_many0(
        pair(space_delimited(alt((char('*'), char('/')))), factor),
        move || init.clone(),
        |acc, (op, val): (char, Expression)| {
            let span = calc_offset(i, acc.span);
            match op {
                '*' => Expression::new(ExprEnum::Mul(Box::new(acc), Box::new(val)), span),
                '/' => Expression::new(ExprEnum::Div(Box::new(acc), Box::new(val)), span),
                _ => panic!(
                    "Multiplicative expression should have '*' \
            or '/' operator"
                ),
            }
        },
    )(r);
    res
}

fn bool_expr(b: Span) -> IResult<Span, Expression> {
    bool_literal(b)
}

fn num_expr(i: Span) -> IResult<Span, Expression> {
    let (r, init) = term(i)?;

    let res = fold_many0(
        pair(space_delimited(alt((char('+'), char('-')))), term),
        move || init.clone(),
        |acc, (op, val): (char, Expression)| {
            let span = calc_offset(i, acc.span);
            match op {
                '+' => Expression::new(ExprEnum::Add(Box::new(acc), Box::new(val)), span),
                '-' => Expression::new(ExprEnum::Sub(Box::new(acc), Box::new(val)), span),
                _ => panic!("Additive expression should have '+' or '-' operator"),
            }
        },
    )(r);
    res
}

fn cond_expr(i0: Span) -> IResult<Span, Expression> {
    let (i, first) = num_expr(i0)?;
    let (i, cond) = space_delimited(alt((char('<'), char('>'))))(i)?;
    let (i, second) = num_expr(i)?;
    let span = calc_offset(i0, i);
    Ok((
        i,
        match cond {
            '<' => Expression::new(ExprEnum::Lt(Box::new(first), Box::new(second)), span),
            '>' => Expression::new(ExprEnum::Gt(Box::new(first), Box::new(second)), span),
            _ => unreachable!(),
        },
    ))
}

fn open_brace(i: Span) -> IResult<Span, ()> {
    let (i, _) = space_delimited(char('{'))(i)?;
    Ok((i, ()))
}

fn close_brace(i: Span) -> IResult<Span, ()> {
    let (i, _) = space_delimited(char('}'))(i)?;
    Ok((i, ()))
}

fn if_expr(i0: Span) -> IResult<Span, Expression> {
    let (i, _) = space_delimited(tag("if"))(i0)?;
    let (i, cond) = expr(i)?;
    let (i, t_case) = delimited(open_brace, statements, close_brace)(i)?;
    let (i, f_case) = opt(preceded(
        space_delimited(tag("else")),
        alt((
            delimited(open_brace, statements, close_brace),
            map_res(
                if_expr,
                |v| -> Result<Vec<Statement>, nom::error::Error<&str>> {
                    Ok(vec![Statement::Expression(v)])
                },
            ),
        )),
    ))(i)?;

    Ok((
        i,
        Expression::new(
            ExprEnum::If(Box::new(cond), Box::new(t_case), f_case.map(Box::new)),
            calc_offset(i0, i),
        ),
    ))
}

fn expr(i: Span) -> IResult<Span, Expression> {
    alt((if_expr, cond_expr, num_expr, bool_expr))(i)
}

fn var_def(i: Span) -> IResult<Span, Statement> {
    let span = i;
    let (i, td) = space_delimited(type_decl)(i)?;
    let (i, name) = space_delimited(identifier)(i)?;
    let (i, _) = space_delimited(char('='))(i)?;
    let (i, ex) = space_delimited(expr)(i)?;
    let (i, _) = space_delimited(char(';'))(i)?;
    Ok((
        i,
        Statement::VarDef {
            span: calc_offset(span, i),
            name,
            td,
            ex,
            storage: false,
        },
    ))
}

fn var_assign(i: Span) -> IResult<Span, Statement> {
    println!("var_assign {:?}", i.fragment());
    let span = i;
    let (i, name) = space_delimited(identifier)(i)?;
    let (i, _) = space_delimited(char('='))(i)?;
    let (i, ex) = space_delimited(expr)(i)?;
    let (i, _) = space_delimited(char(';'))(i)?;
    Ok((
        i,
        Statement::VarAssign {
            span: calc_offset(span, i),
            name,
            ex,
        },
    ))
}

fn expr_statement(i: Span) -> IResult<Span, Statement> {
    let (i, res) = expr(i)?;
    Ok((i, Statement::Expression(res)))
}

fn type_decl(i: Span) -> IResult<Span, TypeDecl> {
    let (i, td) = space_delimited(identifier)(i)?;
    Ok((
        i,
        match *td.fragment() {
            "uint256" => TypeDecl::Uint256,
            "string" => TypeDecl::Str,
            "bool" => TypeDecl::Bool,
            c => {
                println!("type_decl: {:?}", c);
                return Err(nom::Err::Failure(nom::error::Error::new(
                    td,
                    nom::error::ErrorKind::Verify,
                )));
            }
        },
    ))
}

fn argument(i: Span) -> IResult<Span, (Span, TypeDecl)> {
    let (i, td) = type_decl(i)?;
    let (i, ident) = space_delimited(identifier)(i)?;

    Ok((i, (ident, td)))
}

fn fn_def_statement(i: Span) -> IResult<Span, Statement> {
    let (i, _) = space_delimited(tag("function"))(i)?;
    let (i, (name, args, ret_type, stmts)) = cut(|i| {
        let (i, name) = space_delimited(identifier)(i)?;
        let (i, _) = space_delimited(tag("("))(i)?;
        let (i, args) = separated_list0(char(','), space_delimited(argument))(i)?;
        let (i, _) = space_delimited(tag(")"))(i)?;
        let (i, ret_type) = opt(fn_return_def)(i)?;
        let (i, stmts) = delimited(open_brace, statements, close_brace)(i)?;
        Ok((i, (name, args, ret_type, stmts)))
    })(i)?;
    Ok((
        i,
        Statement::FnDef {
            name,
            args,
            ret_type,
            stmts,
        },
    ))
}

fn fn_return_def(i: Span) -> IResult<Span, TypeDecl> {
    let (i, _) = space_delimited(tag("returns"))(i)?;
    let (i, _) = space_delimited(tag("("))(i)?;
    let (i, td) = type_decl(i)?;
    let (i, _) = space_delimited(tag(")"))(i)?;

    Ok((i, td))
}

fn return_statement(i: Span) -> IResult<Span, Statement> {
    let (i, _) = space_delimited(tag("return"))(i)?;
    let (i, ex) = space_delimited(expr)(i)?;
    Ok((i, Statement::Return(ex)))
}

fn general_statement<'a>() -> impl Fn(Span<'a>) -> IResult<Span<'a>, Statement> {
    let terminator = move |i| -> IResult<Span, ()> {
        let mut semicolon = pair(tag(";"), multispace0);
        Ok((semicolon(i)?.0, ()))
    };
    move |input| {
        alt((
            fn_def_statement,
            terminated(return_statement, terminator),
            terminated(expr_statement, terminator),
            var_assign,
            var_def,
        ))(input)
    }
}

pub fn statement(input: Span) -> IResult<Span, Statement> {
    general_statement()(input)
}

pub fn statements(i: Span) -> IResult<Span, Statements> {
    let (i, stmts) = many0(statement)(i)?;
    let (i, _) = opt(multispace0)(i)?;

    Ok((i, stmts))
}

fn contract(i: Span) -> IResult<Span, Statements> {
    let (i, _) = space_delimited(tag("contract"))(i)?;
    let (i, _) = space_delimited(identifier)(i)?;
    delimited(open_brace, statements, close_brace)(i)
}

pub fn statements_finish(i: Span) -> Result<Statements, nom::error::Error<Span>> {
    let (_, res) = contract(i).finish()?;
    Ok(res)
}
