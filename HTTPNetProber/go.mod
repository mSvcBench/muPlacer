module github.com/msvcbench/netprober
require (
    github.com/msvcbench/netprober/client v0.0.0 
    github.com/msvcbench/netprober/responder v0.0.0
)
replace github.com/msvcbench/netprober/client v0.0.0 => ./internal/client
replace github.com/msvcbench/netprober/responder v0.0.0 => ./internal/responder

go 1.21.4
