#!/usr/bin/bash


curl -X POST http://localhost:49112/tts   -H 'Content-Type: application/json'   -d '{"input":"hello from docker","voice":"default.wav","response_format":"wav"}'   --output ./test.wav