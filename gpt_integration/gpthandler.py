import os

import openai

import json

items = ["apples", "bananas", "oranges", "lemons", "pears"]
item_list_string = ', '.join(items)

def get_linear_relationship(w_1, w_2, rel):
    """
        GPT function that gets a linear relationship between the values of different sets.

        Args:
            w1, w2 (lists): Sets that we want to get a relationship between
            rel (string): Relationship of the form >=, <=, =

        Returns:
            (dict): Dictionary representing the two sets and their relationship.
    """
    ordering_info = {
        "items" : items,
        #"values" : ["value of one apple", "value of one pear"],
        "w_1": w_1,
        "w_2": w_2,
        "relationship": rel,
    }
    return json.dumps(ordering_info)

def validate_function(function):
    """
        Validate that the function is accepting/returning the correct object types

        Args:
            function (function): function to validate
    """
    
    assert "name" in function and isinstance(
        function["name"], str
    ), "'name' must be a string."
    assert "description" in function and isinstance(
        function["description"], str
    ), "'description' must be a string."
    assert "parameters" in function and isinstance(
        function["parameters"], dict
    ), "'parameters' must be a dictionary."

    # Check the structure of 'parameters' key
    params = function["parameters"]

    assert (
        "type" in params and params["type"] == "object"
    ), "'type' must be 'object' in parameters."
    assert "properties" in params and isinstance(
        params["properties"], dict
    ), "'properties' must be a dictionary."
    assert "required" in params and isinstance(
        params["required"], list
    ), "'required' must be a list."

    # Check the structure of 'properties' in 'parameters'
    for key, prop in params["properties"].items():
        assert "type" in prop and isinstance(
            prop["type"], str
        ), f"'type' must be a string in properties of {key}."

        if prop["type"] == "array":
            assert (
                "items" in prop
            ), f"'items' must be present in properties of {key} when type is 'array'."

        # Enum check only if it exists
        if "enum" in prop:
            assert isinstance(
                prop["enum"], list
            ), f"'enum' must be a list in properties of {key}."

    # Check 'required' properties are in 'properties'
    for key in params["required"]:
        assert (
            key in params["properties"]
        ), f"'{key}' mentioned in 'required' must exist in 'properties'."

def get_preference_relationship(message, items):
    """
        Get a preference relationship given a message from the human user using GPT's function calling API

        Args:
            message (string): message from human user
            items (list): List of item categories.
    """
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": message}, {"role": "user", "content": "Give me a linear relationship that represents my preferences using only the provided function. If no values are provided for a given item, assume the value is 1 (ex. Pears over Oranges is the same as 1 pear over 1 orange). If an item is not mentioned, assume its value is 0"},{"role": "user", "content": "Make sure that the items in w_1 and w_2 are in the same order."},{"role": "user", "content": "Answer even if you are unsure."}]

    functions = [
        {
            "name": "get_linear_relationship",
            "description": "Get a linear relationship between the values of different sets",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                            "items": {
                                "type": "string",
                            }, 
                        "description": "A list of items."
                    },
                    "values": {
                        "type": "array",
                            "items": {
                                "type": "number",
                            }, 
                        "description": "A array that represents the value of items for the user.",
                    },
                    "w_1": {
                        "type": "array",
                        "items": {
                            "type": "number",
                        },  
                        "description": "Ordered list. The value of "+ items + " in the first set.",
                    },
                    "w_2": {
                        "type": "array",
                        "items": {
                            "type": "number",
                        },                          
                        "description": "Ordered list. The value of " + items + " in the second set.",
                    },
                    "rel": {
                        "type": "string", "enum": [">=", "=<"],
                        "description": "The relationship between the total values of two sets. rel is >= if w_1*values >= w_2*values, and is <= if w_1*values <= w_2*values"
                    },
                },
                "required": ["items", "w_1", "w_2", "rel"],
            },
        }
    ]
    # Validate the function
    validate_function(functions[0])


    # Obtain a response from GPT using the function
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )

    # Return the output
    response_message = response["choices"][0]["message"]
    return response_message