import pytest
import requests
import json

# Define URL and audio files
URL = 'https://irfansharif--diarization-model-web.modal.run'
AUDIO_FILES = {
    'short': 'wavs/short.wav',
    'preamble': 'wavs/preamble.wav',
    'long': 'wavs/long.wav'
}

SCENARIOS = [  # (audio_type, batch_size, assisted)
    ('short', '24', 'false'),
    ('short', '1', 'true'),
    ('long', '24', 'false'),
    ('long', '1', 'true'),
    ('preamble', '24', 'false'),
    ('preamble', '1', 'true'),
]

# Fixture to perform HTTP POST request
@pytest.fixture
def perform_request(benchmark):

    def _perform_request(audio_type, batch_size, assisted):
        def fn():
            files = {'file': (AUDIO_FILES[audio_type], open(AUDIO_FILES[audio_type], 'rb'), 'audio/wav')}
            parameters = {'batch_size': batch_size, 'assisted': assisted}
            data = {'parameters': json.dumps(parameters)}
            return requests.post(URL, files=files, data=data)
        return benchmark(fn)

    return _perform_request

# Benchmark test function
@pytest.mark.parametrize("audio_type, batch_size, assisted", SCENARIOS)
def test_load(perform_request, audio_type, batch_size, assisted):
    result = perform_request(audio_type, batch_size, assisted)
    assert result.status_code == 200  # Assert successful response
    print(json.dumps(result.json(), indent=4))  # print response

# curl -X POST 'https://irfansharif--diarization-model-web.modal.run' \
#     -F 'file=@wavs/short.wav;type=audio/wav' \
#     -F 'parameters={"batch_size":24,"assisted":false}'



