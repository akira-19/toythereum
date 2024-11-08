# toythereum

toythereum has the toy solidity compiler and evm.
It's not fully functioned like solc and geth, but it can compile solidity like language and you can run the bytecode in the evm.

## how to use

1. write your solidity code, and put the file into contracts dir
   but it accept only uint256, bool, and string so far.
   Additionally, it doesn't have public, private, view, and etc..
   see example.sol.

2. compile your contract

```
$ sh cmd/compile_contract.sh
```

# info from ethereum yellow paper

## 9.1 evm basics

- stack based architecture
- stack max size: 1024
- word size: 256bit
- memory model: word-addressed byte array
- storage model: word-addressed word array
- not following von Neumann architecture
- program code is stored separately in a virtual ROM

## 9.3 Execution Environment

necessary information to change the state

- current state: σ
- remaining gas for computation: g
- other information I
  - I_a: contract address
  - I_o: original sender address
  - I_p: gas price
  - I_d: input data byte array
  - I_s: sender address
  - I_v: value in wei
  - I_H: the block header of the parent block
  - I_e: the depth of the present message-call or contract creation
  - I_w: the permission to make modifications to the state
