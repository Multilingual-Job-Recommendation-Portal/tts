artifacts:
	wget https://ai4b-public-nlu-nlg.objectstore.e2enetworks.net/en2indic.zip 
	unzip en2indic.zip
	rm en2indic.zip
	mv en-indic ./models/indic-trans/en-indic 
	wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/as.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/bn.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/brx.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/en.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/gu.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/hi.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/kn.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/ml.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/mni.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/mr.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/or.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/pa.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/raj.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/ta.zip https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/te.zip
	mv *.zip ./models/v1/	 
	unzip '*.zip'
	rm *.zip
	
requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

install:
	pip install -r requirements-ml.txt 
	pip install -r requirements-utils.txt 
	pip install -r requirements-server.txt 

environment:
	python3 -m venv .venv
	. .venv/bin/activate

setup:
	make artifacts
	make environment
	make install
	pre-commit install

build:
	uvicorn main:app --reload

documentation:
	poetry add pdoc
	pdoc y4j_recommender -o docs/html -d google --footer-text "Y4J Recommendation Engine"

tunnel:
	ngrok http 8000
