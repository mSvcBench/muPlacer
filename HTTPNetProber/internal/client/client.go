package client

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/golang/glog"
)

func Client(w http.ResponseWriter, r *http.Request) {

	duration_string := r.URL.Query().Get("duration") // duration of the test
	dest_url := r.URL.Query().Get("url")             // destination URL
	chunk_str := r.URL.Query().Get("chunk")          // chunk size_str of the server write buffer
	size_str := r.URL.Query().Get("size")            // size of the server response, used only when duration is 0
	client := http.Client{}
	total_bytes := 0
	buf := make([]byte, 256*1024) //the chunk size

	// Check if the duration parameter is empty
	if duration_string == "" {
		duration_string = "0" // default duration string
	}

	// Check if the dest parameter is empty
	if dest_url == "" {
		dest_url = "http://127.0.0.1:8080" // default destination url
	}

	// Check if the dest parameter is empty
	if chunk_str == "" {
		chunk_str = strconv.Itoa(256 * 1024) // default chunk size
	}

	// Check if the size parameter is empty
	if size_str == "" {
		size_str = "0" // default size, used for ping
	}

	// Make an HTTP request to the destination URL
	full_dest_url := dest_url + "/generate?duration=" + duration_string + "&chunk=" + chunk_str + "&size=" + size_str
	glog.Info("Making request to: " + full_dest_url + " for " + duration_string + "s" + " with chunk size " + chunk_str + "bytes and message size " + size_str + " bytes")
	req, _ := http.NewRequest("GET", full_dest_url, nil)
	start := time.Now()
	res, err := client.Do(req)
	if err != nil {
		// Handle error
		glog.Errorf("Error making HTTP request: %s", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}
	rd := bufio.NewReader(res.Body)
	for {
		n, err := rd.Read(buf) //loading chunk into buffer
		total_bytes += n
		if n == 0 {
			if err == io.EOF {
				break
			}
			if err != nil {
				glog.Errorf("Error making HTTP request: %s", err)
				break
			}

		}
	}
	res.Body.Close()
	total_duration := time.Since(start).Seconds()

	// Create a JSON object with keys total_bytes and total_duration
	result := map[string]any{
		"total_bytes":    total_bytes,
		"total_duration": total_duration,
		"total_speed":    float64(total_bytes) * 8.0 / total_duration,
	}
	// Convert the JSON object to a string
	jsonString, _ := json.Marshal(result)

	// Set the response content type to application/json
	w.Header().Set("Content-Type", "application/json")

	// Write the JSON string as the response body
	fmt.Fprint(w, string(jsonString))
}
