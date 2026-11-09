data:
	python -m src.data
train:
	python -m src.train
test:
	pytest -q
score:
	python -m src.score --input sample.csv --output scored.csv
