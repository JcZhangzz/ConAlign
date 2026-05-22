# ConAlign

## Environment

- Python 3.8.0

## Run ConAlign

You can run the main model with:

```bash
python3 mymodel.py --dataset coat
```

Supported datasets:

- `coat`
- `yahooR3`
- `KuaiRand`

Examples:

```bash
python3 mymodel.py --dataset coat
python3 mymodel.py --dataset yahooR3
python3 mymodel.py --dataset KuaiRand
```

## Run Baselines

Example:

```bash
python3 baselines/InterD.py --dataset coat
```

You can also replace `coat` with other supported datasets:

```bash
python3 baselines/InterD.py --dataset yahooR3
python3 baselines/InterD.py --dataset KuaiRand
```
