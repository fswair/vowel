from vowel.monitoring import enable_monitoring
from vowel.runner import Function, RunEvals

enable_monitoring(
    logfire_enabled=True,
    service_name="quality-judge",
)

runner = RunEvals.from_file("largest_color_value_judge.yml")

main_runner = runner.with_serializer({"evals.generate_spec": Function}).filter(
    "evals.generate_spec"
)

# mock_runner = runner.with_serializer({"evals.generate_spec_mock": Function}).filter(
#     "evals.generate_spec_mock"
# )


summary = main_runner.run()

summary.print()
