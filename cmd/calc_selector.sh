#!/bin/bash

DIR="$(cd "$(dirname "$0")"/.. && pwd)"
"$DIR/target/debug/selector" "$1"
