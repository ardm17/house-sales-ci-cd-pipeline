install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

train:
	python train.py

eval:
	echo "##Model Metrics" > report.md && \
	cat ./Results/metrics.txt >> report.md && \
	echo '\n## Confusion Matrix Plot' >> report.md && \
	cml comment create report.md

update branch:
	git config --global http.postBuffer 524288000 && \
	git config --global user.name $(USER_NAME) && \
	git config --global user.email $(USER_EMAIL) && \
	git commit -am "Update with new results" && \
	git push --force origin HEAD:update

deploy:
	huggingface-cli login --token $(HF) && \
	huggingface-cli upload housesale ./App --repo-type=space --commit-message="Update App" && \
	huggingface-cli upload housesale ./Model --repo-type=space --commit-message="Update Model"