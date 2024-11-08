#!/bin/bash

DIR="$(cd "$(dirname "$0")"/.. && pwd)"
"$DIR/target/debug/compiler" "$DIR/contracts/$1"
