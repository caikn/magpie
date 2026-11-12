package gateway

import (
	"encoding/json"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestGeminiAtomicSignaturesStayWithTheirProvider(t *testing.T) {
	fresh(t)
	const thoughtSig, textSig = "atomic-thought-signature", "atomic-text-signature"
	var decoder codeAssistDecoder
	var col collector
	decode := signersNoted(decoder.decode, "google-vertex")
	if err := decode(`{"candidates":[{"content":{"role":"model","parts":[{"text":"Think.","thought":true,"thoughtSignature":"`+thoughtSig+`"},{"text":"Answer.","thoughtSignature":"`+textSig+`"}]},"finishReason":"STOP"}]}`, col.add); err != nil {
		t.Fatal(err)
	}
	for _, sig := range []string{thoughtSig, textSig} {
		if got := signerOf(sig); got != "google-vertex" {
			t.Errorf("atomic signature %q signer = %q, want google-vertex", sig, got)
		}
	}
	req := &Request{Messages: []Message{{Role: "assistant", Parts: col.finish().Parts}}}
	if got := ownSignatures(req, "google-vertex"); !reflect.DeepEqual(got, req) {
		t.Error("the signer's own history changed")
	}
	for _, part := range ownSignatures(req, "google-aistudio-paid").Messages[0].Parts {
		if part.Signature != "" {
			t.Errorf("foreign %v part kept signature %q", part.Kind, part.Signature)
		}
	}
	claude := &Request{Messages: []Message{{Role: "assistant", Parts: []Part{{Kind: Thinking, Signature: thoughtSig, SignedBy: sealAnthropic}}}}}
	if got := ownSignatures(claude, "google-aistudio-paid"); !reflect.DeepEqual(got, claude) {
		t.Error("Claude's signed thinking changed")
	}
}

func TestGeminiSignedPartsToolHistory(t *testing.T) {
	const signed = `[{"text":"Check tool.","thought":true,"thoughtSignature":"thought-signature"},{"text":"Use tool.","thoughtSignature":"text-signature"},{"text":"","thoughtSignature":"empty-signature"},{"functionCall":{"id":"multiply-1","name":"multiply","args":{"x":19,"y":23}},"thoughtSignature":"call-signature"}]`
	var expected []any
	if err := json.Unmarshal([]byte(signed), &expected); err != nil {
		t.Fatal(err)
	}
	for _, stream := range []bool{false, true} {
		var decoder codeAssistDecoder
		var col collector
		rec := httptest.NewRecorder()
		enc := geminiEncoder{w: newSSEWriter(rec), model: "gemini-3.8-flash"}
		tidy := textCallSee{names: map[string]bool{"multiply": true}, see: func(ev Event) {
			col.add(ev)
			enc.event(ev)
		}}
		emit := tidy.event
		if err := decoder.decode(`{"candidates":[{"content":{"role":"model","parts":`+signed+`},"finishReason":"STOP"}]}`, emit); err != nil {
			t.Fatal(err)
		}
		enc.finish()
		var parts []any
		if stream {
			for _, line := range strings.Split(rec.Body.String(), "\n") {
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				var event struct {
					Candidates []struct {
						Content struct {
							Parts []any `json:"parts"`
						} `json:"content"`
					} `json:"candidates"`
				}
				if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
					t.Fatal(err)
				}
				for _, c := range event.Candidates {
					parts = append(parts, c.Content.Parts...)
				}
			}
		} else {
			var reply struct {
				Candidates []struct {
					Content struct {
						Parts []any `json:"parts"`
					} `json:"content"`
				} `json:"candidates"`
			}
			if err := json.Unmarshal(renderGemini(col.finish(), "gemini-3.8-flash"), &reply); err != nil {
				t.Fatal(err)
			}
			parts = reply.Candidates[0].Content.Parts
		}
		if !reflect.DeepEqual(parts, expected) {
			t.Fatalf("stream=%v: signed native parts changed: got %#v, want %#v", stream, parts, expected)
		}
		model, _ := json.Marshal(parts)
		request, err := parseGemini([]byte(`{"model":"gemini-3.8-flash","contents":[{"role":"user","parts":[{"text":"multiply"}]},{"role":"model","parts":` + string(model) + `},{"role":"user","parts":[{"functionResponse":{"id":"multiply-1","name":"multiply","response":{"product":437}}}]}]}`))
		if err != nil {
			t.Fatal(err)
		}
		var sent struct {
			Request struct {
				Contents []struct {
					Role  string `json:"role"`
					Parts []any  `json:"parts"`
				} `json:"contents"`
			} `json:"request"`
		}
		if err := json.Unmarshal(buildCodeAssist(request, "gemini-3.8-flash", ""), &sent); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(sent.Request.Contents[1].Parts, expected) {
			t.Fatalf("stream=%v: signed tool history changed: %#v", stream, sent.Request.Contents[1].Parts)
		}
	}
}
