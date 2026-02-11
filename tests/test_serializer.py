"""Tests for with_serializer (schema and serial_fn)."""

from pydantic import BaseModel

from vowel import RunEvals


class User(BaseModel):
    id: int
    name: str
    email: str


class Config(BaseModel):
    timeout: int
    verbose: bool


def get_user_info(user: User) -> str:
    return f"User {user.name} has email {user.email}"


def compare_users(user1: User, user2: User) -> bool:
    return user1.id < user2.id


def process_with_config(user: User, config: Config) -> str:
    return f"{user.name} (timeout={config.timeout})"


class TestSchemaSerializer:
    """Tests for schema-based serialization."""

    def test_input_dict_to_model(self):
        """input: dict → Model (single param)"""
        spec = {
            "get_user_info": {
                "dataset": [
                    {
                        "case": {
                            "input": {"id": 1, "name": "Alice", "email": "a@a.com"},
                            "expected": "User Alice has email a@a.com",
                        }
                    },
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"get_user_info": get_user_info})
            .with_serializer({"get_user_info": User})
            .run()
        )
        assert summary.all_passed

    def test_inputs_dict_to_model(self):
        """inputs: dict → Model (single param using inputs)"""
        spec = {
            "get_user_info": {
                "dataset": [
                    {
                        "case": {
                            "inputs": {"id": 2, "name": "Bob", "email": "b@b.com"},
                            "expected": "User Bob has email b@b.com",
                        }
                    },
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"get_user_info": get_user_info})
            .with_serializer({"get_user_info": User})
            .run()
        )
        assert summary.all_passed

    def test_inputs_single_element_list(self):
        """inputs: [dict] → Model (single element list)"""
        spec = {
            "get_user_info": {
                "dataset": [
                    {
                        "case": {
                            "inputs": [{"id": 3, "name": "Charlie", "email": "c@c.com"}],
                            "expected": "User Charlie has email c@c.com",
                        }
                    },
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"get_user_info": get_user_info})
            .with_serializer({"get_user_info": User})
            .run()
        )
        assert summary.all_passed

    def test_inputs_multi_element_list_same_type(self):
        """inputs: [dict, dict] → Model, Model (multiple params, same type)"""
        spec = {
            "compare_users": {
                "dataset": [
                    {
                        "case": {
                            "inputs": [
                                {"id": 1, "name": "Alice", "email": "a@a.com"},
                                {"id": 2, "name": "Bob", "email": "b@b.com"},
                            ],
                            "expected": True,
                        }
                    },
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"compare_users": compare_users})
            .with_serializer({"compare_users": User})
            .run()
        )
        assert summary.all_passed

    def test_inputs_named_params_different_types(self):
        """inputs: {param: dict} → different types with dict schema"""
        spec = {
            "process_with_config": {
                "dataset": [
                    {
                        "case": {
                            "inputs": {
                                "user": {"id": 1, "name": "Alice", "email": "a@a.com"},
                                "config": {"timeout": 30, "verbose": True},
                            },
                            "expected": "Alice (timeout=30)",
                        }
                    },
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"process_with_config": process_with_config})
            .with_serializer({"process_with_config": {"user": User, "config": Config}})
            .run()
        )
        assert summary.all_passed

    def test_no_serializer_passthrough(self):
        """Without serializer, dict is passed as-is."""

        def greet(name: str) -> str:
            return f"Hello {name}"

        spec = {
            "greet": {
                "dataset": [
                    {"case": {"input": "World", "expected": "Hello World"}},
                ]
            }
        }
        summary = RunEvals.from_dict(spec).with_functions({"greet": greet}).run()
        assert summary.all_passed

    def test_multiple_cases(self):
        """Multiple cases with schema serializer."""
        spec = {
            "get_user_info": {
                "dataset": [
                    {
                        "case": {
                            "input": {"id": 1, "name": "Alice", "email": "a@a.com"},
                            "expected": "User Alice has email a@a.com",
                        }
                    },
                    {
                        "case": {
                            "input": {"id": 2, "name": "Bob", "email": "b@b.com"},
                            "expected": "User Bob has email b@b.com",
                        }
                    },
                    {
                        "case": {
                            "input": {"id": 3, "name": "Charlie", "email": "c@c.com"},
                            "expected": "User Charlie has email c@c.com",
                        }
                    },
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"get_user_info": get_user_info})
            .with_serializer({"get_user_info": User})
            .run()
        )
        assert summary.all_passed
        assert summary.total_count == 1


class TestSerialFn:
    """Tests for serial_fn-based serialization."""

    def test_serial_fn_basic(self):
        """serial_fn receives raw dict and returns serialized value."""

        def serialize_user(d: dict) -> User:
            data = d.get("input") or d.get("inputs")
            assert data is not None
            return User(**data)

        spec = {
            "get_user_info": {
                "dataset": [
                    {
                        "case": {
                            "inputs": {"id": 1, "name": "Alice", "email": "a@a.com"},
                            "expected": "User Alice has email a@a.com",
                        }
                    },
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"get_user_info": get_user_info})
            .with_serializer(serial_fn={"get_user_info": serialize_user})
            .run()
        )
        assert summary.all_passed

    def test_serial_fn_returns_tuple(self):
        """serial_fn returning tuple unpacks as positional args."""

        def serialize_users(d: dict) -> tuple:
            inputs = d["inputs"]
            return (User(**inputs[0]), User(**inputs[1]))

        spec = {
            "compare_users": {
                "dataset": [
                    {
                        "case": {
                            "inputs": [
                                {"id": 1, "name": "Alice", "email": "a@a.com"},
                                {"id": 2, "name": "Bob", "email": "b@b.com"},
                            ],
                            "expected": True,
                        }
                    },
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"compare_users": compare_users})
            .with_serializer(serial_fn={"compare_users": serialize_users})
            .run()
        )
        assert summary.all_passed

    def test_serial_fn_returns_dict(self):
        """serial_fn returning dict unpacks as keyword args."""

        def serialize_process(d: dict) -> dict:
            inputs = d["inputs"]
            return {
                "user": User(**inputs["user"]),
                "config": Config(**inputs["config"]),
            }

        spec = {
            "process_with_config": {
                "dataset": [
                    {
                        "case": {
                            "inputs": {
                                "user": {"id": 1, "name": "Alice", "email": "a@a.com"},
                                "config": {"timeout": 30, "verbose": True},
                            },
                            "expected": "Alice (timeout=30)",
                        }
                    },
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"process_with_config": process_with_config})
            .with_serializer(serial_fn={"process_with_config": serialize_process})
            .run()
        )
        assert summary.all_passed

    def test_serial_fn_full_control(self):
        """serial_fn has full control over input processing."""

        def custom_serializer(d: dict) -> User:
            data = d.get("input") or d.get("inputs")
            assert data is not None
            return User(
                id=data["id"],
                name=data["first_name"] + " " + data["last_name"],
                email=data["email"],
            )

        def get_full_name(user: User) -> str:
            return user.name

        spec = {
            "get_full_name": {
                "dataset": [
                    {
                        "case": {
                            "input": {
                                "id": 1,
                                "first_name": "John",
                                "last_name": "Doe",
                                "email": "j@d.com",
                            },
                            "expected": "John Doe",
                        }
                    },
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"get_full_name": get_full_name})
            .with_serializer(serial_fn={"get_full_name": custom_serializer})
            .run()
        )
        assert summary.all_passed


class TestSerializerChaining:
    """Tests for serializer method chaining."""

    def test_chaining_with_other_methods(self):
        """with_serializer chains with other RunEvals methods."""
        spec = {
            "get_user_info": {
                "dataset": [
                    {
                        "case": {
                            "input": {"id": 1, "name": "Alice", "email": "a@a.com"},
                            "expected": "User Alice has email a@a.com",
                        }
                    },
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"get_user_info": get_user_info})
            .with_serializer({"get_user_info": User})
            .filter(["get_user_info"])
            .run()
        )
        assert summary.all_passed

    def test_multiple_serializer_calls(self):
        """Multiple with_serializer calls merge."""
        spec = {
            "get_user_info": {
                "dataset": [
                    {
                        "case": {
                            "input": {"id": 1, "name": "Alice", "email": "a@a.com"},
                            "expected": "User Alice has email a@a.com",
                        }
                    },
                ]
            },
            "process_with_config": {
                "dataset": [
                    {
                        "case": {
                            "inputs": {
                                "user": {"id": 1, "name": "Bob", "email": "b@b.com"},
                                "config": {"timeout": 10, "verbose": False},
                            },
                            "expected": "Bob (timeout=10)",
                        }
                    },
                ]
            },
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions(
                {"get_user_info": get_user_info, "process_with_config": process_with_config}
            )
            .with_serializer(
                {"get_user_info": User, "process_with_config": {"user": User, "config": Config}}
            )
            .run()
        )
        assert summary.all_passed


class TestSerializerEdgeCases:
    """Edge cases for serializer."""

    def test_serializer_with_callable(self):
        """Schema can be any callable, not just types."""
        from datetime import date

        def parse_date(s: str) -> date:
            return date.fromisoformat(s)

        def get_year(d: date) -> int:
            return d.year

        spec = {
            "get_year": {
                "dataset": [
                    {"case": {"input": "2026-01-07", "expected": 2026}},
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"get_year": get_year})
            .with_serializer({"get_year": parse_date})
            .run()
        )
        assert summary.all_passed

    def test_serializer_validation_error(self):
        """Pydantic validation errors surface properly."""
        spec = {
            "get_user_info": {
                "dataset": [
                    {"case": {"input": {"id": 1, "name": "Alice"}, "expected": "anything"}},
                ]
            }
        }
        summary = (
            RunEvals.from_dict(spec)
            .with_functions({"get_user_info": get_user_info})
            .with_serializer({"get_user_info": User})
            .run()
        )
        assert not summary.all_passed
        assert summary.failed_count == 1
