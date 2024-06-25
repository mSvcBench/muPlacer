package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/msvcbench/netprober/client"
	"github.com/msvcbench/netprober/responder"
)

func main() {
	flag.Parse() // parse the flags for logging
	http.HandleFunc("/generate", responder.Responder)
	http.HandleFunc("/get", client.Client)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
