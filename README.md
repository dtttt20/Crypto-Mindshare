# Crypto-Mindshare

Installation and setup instructions for the Crypto-Mindshare project.

## Prerequisites

- Docker
- OpenAI API key
- Openrouter or Perplexity API key

## Installation

1. Clone the repository
2. Navigate to the `docker` directory
3. Create a `.env` file based on the `.env.example` file
4. Run ```docker compose up --build```

## Configuration

### Tweet Database

There are two options for the database:

1. Use the sample tweet database for testing
2. Use your own database

To select the database, uncomment the section within the .env and
the docker-compose.yaml file that you want to use.

#### User Config

The user config section is used to configure the project for the user. It is set in your .env file in the `docker` directory.

###### OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
```

######Perplexity API key or Openrouter API key:
```
PERPLEXITY_API_KEY=your_perplexity_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```
Uncomment whichever API key you want to use.

The Openrouter option is used to access the Perplexity API. It is provided
because the Perplexity API does not allow structured outputs for anyone below tier 3
(which requires $500 to have been spent on the API).
> [!NOTE]
    > The Perplexity API is expensive and processing lots of tweets will quickly add up.

###### Starting Timestamp:
```
STARTING_TIMESTAMP=your_starting_timestamp
```
The starting timestamp is used to start processing tweets from a specific point in time. It is also used to resume processing from the last processed tweet if the project is stopped and restarted.
- The default is set to 2025-02-24T11:00:00Z.
- The sample tweet database has tweets from 2025-03-03 20:41:14Z to 2025-03-04 21:19:27Z

> [!NOTE]
    > If you set this to an early timestamp, it will use a lot of tokens and be very expensive due to the checking of the project and token existence.

###### Maximum number of tweets to process per batch:
```
MAX_TWEETS_PER_BATCH=200
```
Each time the mindshare processor is run, it will process the number of tweets specified by MAX_TWEETS_PER_BATCH.
- Default is 200.

###### Maximum number of concurrent tweets to process:
```
MAX_CONCURRENT_TWEETS=200
```
This is the maximum number of tweets that will be processed concurrently.
- Default is 200.

###### Maximum number of concurrent validations to perform:
```
MAX_CONCURRENT_VALIDATIONS=70
```
This is the maximum number of validations that will be performed concurrently.
- Default is 70.

###### Maximum number of concurrent OpenAI API calls allowed:
```
OPENAI_MAX_CONCURRENT_REQUESTS=4900
TIME_WINDOW_SECONDS=60
```
This is the maximum number of OpenAI API calls that your OpenAI API key allows.
- Default is 4900.

###### Maximum number of concurrent Openrouter API calls allowed:
```
OPENROUTER_MAX_CONCURRENT_REQUESTS=65
TIME_WINDOW_SECONDS=10
```
This is the maximum number of Openrouter API calls that your Openrouter API key allows.
- Default is 65.

###### Maximum number of concurrent perplexity api calls to perform:
```
PERPLEXITY_MAX_CONCURRENT_REQUESTS=60
TIME_WINDOW_SECONDS=60
```
This is the maximum number of perplexity API calls that your perplexity API key allows.
- Default is 60.

### docker-compose.yaml

The `docker-compose.yaml` file is used to start the project. It is located in the `docker` directory.

## Running the project

1. Run ```docker compose up --build```
2. Run ```docker compose down``` to stop the project

## Architecture

