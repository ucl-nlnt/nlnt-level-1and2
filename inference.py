from simple_chalk import chalk
import requests
import json
import argparse
import time


def main(username, prompt):
    print(chalk.green("UCL NLNT Level 1 and 2 Inference Terminal"))

    url = 'http://10.158.18.253:8000/send-prompt'
    # url = 'http://localhost:8000/send-prompt'

    # supply username/prompt if they are not provided
    #if username is None:
    #    username = input(chalk.yellow(
    #        "Enter your username (red,gab,mara,rica): "))
    if prompt is None:
        prompt = input(chalk.yellow("Enter your prompt: "))

    # TODO: format prompt

    headers = {'Content-Type': 'application/json'}
    data = {'content': prompt}

    try:
        print(chalk.magenta("sending prompt to inference server..."))
        start_time = time.time()

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(chalk.magenta("response received!"),
              f'{round(elapsed_time, 2)} seconds')

        print(response)
        
        json_object = response.json()
        json_formatted_str = json.dumps(json_object, indent=2)
        print(json_formatted_str)
    except requests.RequestException as e:
        print("Error:", e)
        return None

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--username', type=str,
                        help='Your username', required=False)
    parser.add_argument('--prompt', type=str,
                        help='Your prompt', required=False)
    args = parser.parse_args()
    main(args.username, args.prompt)
