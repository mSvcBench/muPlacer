package client

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"
)

func Client(w http.ResponseWriter, r *http.Request) {

	var s, d int

	length := r.URL.Query().Get("length")
	duration := r.URL.Query().Get("duration")
	dest_url := r.URL.Query().Get("url")
	client := http.Client{}

	// Check if the length parameter is empty
	if length == "" {
		s = 1000000 // default byte size
	} else {
		// println(length)
		s, _ = strconv.Atoi(length)
	}

	// Check if the duration parameter is empty
	if duration == "" {
		d = 60 // default byte size
	} else {
		// println(duration)
		d, _ = strconv.Atoi(duration)
	}

	// Check if the duration parameter is empty
	if dest_url == "" {
		dest_url = "netprobe-server:8080" // default destination url
	}

	// Get the current time
	start := time.Now()
	total_bytes := int(0)
	full_dest_url := dest_url + "/generate?length=" + strconv.Itoa(s)
	if d > 0 {
		for time.Since(start) < time.Duration(d)*time.Second {
			// Make an HTTP request to the destination URL
			fmt.Println("Making request to:", full_dest_url)
			req, _ := http.NewRequest("GET", full_dest_url, nil)
			res, err := client.Do(req)
			if err != nil {
				// Handle error
				fmt.Println("Error making HTTP request:", err)
				http.Error(w, "Internal Server Error", http.StatusInternalServerError)
				return
			}
			body, _ := io.ReadAll(res.Body) // ensure the complete reception of the response
			res.Body.Close()
			// Get the length of the response body
			total_bytes += len(body)
		}
	} else {
		// Make an single HTTP request to the destination URL
		fmt.Println("Making request to:", full_dest_url)
		req, _ := http.NewRequest("GET", full_dest_url, nil)
		res, err := client.Do(req)
		if err != nil {
			// Handle error
			fmt.Println("Error making HTTP request:", err)
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			return
		}
		body, _ := io.ReadAll(res.Body) // ensure the complete reception of the response
		res.Body.Close()
		// Get the length of the response body
		total_bytes += len(body) // Make an HTTP request to the destination URL
	}

	total_duration := time.Since(start).Seconds()
	// Create a JSON object with keys total_bytes and total_duration
	result := map[string]any{
		"total_bytes":    total_bytes,
		"total_duration": total_duration,
	}
	// Convert the JSON object to a string
	jsonString, _ := json.Marshal(result)

	// Set the response content type to application/json
	w.Header().Set("Content-Type", "application/json")

	// Write the JSON string as the response body
	fmt.Fprint(w, string(jsonString))
}
