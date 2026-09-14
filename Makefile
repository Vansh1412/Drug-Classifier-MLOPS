install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:	
	black *.py 

train:
	python train.py

test:
	pytest tests/ -v

docker-build:
	docker build -t drug-classifier .

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	
	cml comment create report.md
		
update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login: 
	pip install -U "huggingface_hub[cli]"
	git pull origin update
	git switch update
	huggingface-cli login --token $(HF) --add-to-git-credential

HF_SPACE ?= Vansh1412/Drug-Classification

push-hub: 
	huggingface-cli upload $(HF_SPACE) ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload $(HF_SPACE) ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload $(HF_SPACE) ./Results /Metrics --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub

all: install format train test docker-build eval update-branch deploy