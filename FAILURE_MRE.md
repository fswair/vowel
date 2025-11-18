### Simulate Failure

```sh
pip install -e .
```

```sh
vowel examples/test_raises.yml

# or

vowel examples/test_raises_reordered.yml
```

You should see unfair difference between second and third case in both example, even though they're same cases with different ids.
