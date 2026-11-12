package gateway

import (
	"encoding/json"
	"testing"
)

func TestGeminiImageToolResultStaysInsideResponse(t *testing.T) {
	for _, tc := range []struct {
		model, agent string
		nested       bool
	}{
		{"gemini-3.8-flash", "", true},
		{"gemini-3.8-flash", "vertex", true},
		{"gemini-4-argon", "gemini", true},
		{"gemini-2.5-flash", "", false},
		{"gemini-3.8-flash", "antigravity", false},
	} {
		t.Run(tc.model+"/"+tc.agent, func(t *testing.T) {
			request, err := parseGemini([]byte(`{"model":"` + tc.model + `","contents":[{"role":"user","parts":[{"text":"Read image 1"}]},{"role":"model","parts":[{"functionCall":{"id":"image-call-1","name":"get_test_image","args":{"index":1}},"thoughtSignature":"synthetic-signature"}]},{"role":"user","parts":[{"functionResponse":{"id":"image-call-1","name":"get_test_image","response":{"description":"image attached"},"parts":[{"inlineData":{"mimeType":"image/png","data":"cG5n"}}]}}]}]}`))
			if err != nil {
				t.Fatal(err)
			}
			var sent struct {
				Request struct {
					Contents []struct {
						Role  string           `json:"role"`
						Parts []map[string]any `json:"parts"`
					} `json:"contents"`
				} `json:"request"`
			}
			if err := json.Unmarshal(buildCodeAssist(request, tc.model, tc.agent), &sent); err != nil {
				t.Fatal(err)
			}
			turn := sent.Request.Contents[len(sent.Request.Contents)-1]
			response := turn.Parts[0]["functionResponse"].(map[string]any)
			if response["id"] != "image-call-1" || response["name"] != "get_test_image" {
				t.Fatalf("image result lost its call identity: %#v", response)
			}
			if tc.nested {
				if len(turn.Parts) != 1 {
					t.Fatalf("native image must remain inside its functionResponse, got %d sibling parts", len(turn.Parts))
				}
				parts, ok := response["parts"].([]any)
				if !ok || len(parts) != 1 {
					t.Fatalf("native functionResponse.parts image missing: %#v", response)
				}
				image := parts[0].(map[string]any)["inlineData"].(map[string]any)
				if image["mimeType"] != "image/png" || image["data"] != "cG5n" {
					t.Fatalf("native image changed: %#v", image)
				}
			} else if len(turn.Parts) != 2 || response["parts"] != nil || turn.Parts[1]["inlineData"] == nil {
				t.Fatalf("legacy image placement changed: %#v", turn.Parts)
			}
		})
	}
}
