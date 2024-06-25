module github.com/msvcbench/netprober

require (
	github.com/msvcbench/netprober/client v0.0.0
	github.com/msvcbench/netprober/responder v0.0.0
)

require github.com/golang/glog v1.2.1 // indirect

replace github.com/msvcbench/netprober/client v0.0.0 => ./internal/client

replace github.com/msvcbench/netprober/responder v0.0.0 => ./internal/responder

go 1.21.4
