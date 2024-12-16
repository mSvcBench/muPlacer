package responder

import (
	"crypto/rand"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/golang/glog"
)

var randString string

func init() {
	randBytes := make([]byte, 256*1024)
	_, err := rand.Read(randBytes)
	if err != nil {
		glog.Info("Failed to generate random string", http.StatusInternalServerError)
		return
	}
	randString = string(randBytes)
}

func Responder(w http.ResponseWriter, r *http.Request) {

	var duration int

	w.Header().Set("Connection", "Keep-Alive")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.Header().Set("X-Content-Type-Options", "nosniff")

	duration_string := r.URL.Query().Get("duration")
	chunk_string := r.URL.Query().Get("chunk")
	size_string := r.URL.Query().Get("size")

	// Check if the dest parameter is empty
	if chunk_string == "" {
		chunk_string = strconv.Itoa(256 * 1024) // default destination url
	}
	chunk, _ := strconv.Atoi(chunk_string)
	chunk = min(chunk, 256*1024)

	// Check if the length parameter is empty
	if duration_string == "" {
		duration = 60 // default duration of 60 seconds
	} else {
		// println(length)
		duration, _ = strconv.Atoi(duration_string)
	}

	if duration > 0 {
		glog.Infof("Responding for %d seconds", duration)
		start := time.Now()
		for time.Since(start) < time.Duration(duration)*time.Second {
			fmt.Fprint(w, randString[:chunk])
			w.(http.Flusher).Flush()
		}
	} else {
		// when duration == 0 return a message with the requestesd size
		size, _ := strconv.Atoi(size_string)
		if size == 0 {
			glog.Info("Responding for ping")
			fmt.Fprint(w, "") // return empty response for http ping
			w.(http.Flusher).Flush()
		} else {
			glog.Infof("Responding with size %d bytes", size)
			for size > 0 {
				fmt.Fprint(w, randString[:min(chunk, size)])
				w.(http.Flusher).Flush()
				size -= chunk
			}
		}
	}
}
