tool_input_raw = "product=laptop ,product=TV"

raw_args = [x.strip() for x in tool_input_raw.split(",")]
args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]


# print(raw_args)
# print(args)


print("       vasudev=tejam        ".split("=", 1)[-1].strip("'\""))
