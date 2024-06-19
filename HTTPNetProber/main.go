package main

import (
	"log"
	"net/http"

	"github.com/msvcbench/netprober/client"
	"github.com/msvcbench/netprober/responder"
)

func main() {
	http.HandleFunc("/generate", responder.Responder)
	http.HandleFunc("/get", client.Client)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
