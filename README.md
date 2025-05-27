# Project: AI UK Travel Guide Chatbot

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Live Demo](#live-demo)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running Locally](#running-locally)
5. [Usage](#usage)
6. [Architecture](#architecture)
   - [System Components](#system-components)
   - [Folder Structure](#folder-structure)
7. [Technical Implementation](#technical-implementation)
   - [NLP & AI Techniques](#nlp--ai-techniques)
   - [API Integration](#api-integration)
   - [User Interface Design](#user-interface-design)
8. [Testing](#testing)
9. [Evaluation & Results](#evaluation--results)
10. [Limitations & Future Work](#limitations--future-work)

---

<h2 id="project-overview">📍 Project Overview</h2>

Planning a trip can feel overwhelming, especially with a flood of travel blogs, booking sites, and endless search results at your fingertips. Imagine if organising a trip to the UK felt more like chatting with a knowledgeable friend. That’s the core idea behind this project. Someone who understands your preferences, knows where to find the best food, and can offer real-time suggestions for places to explore.

My final-year dissertation revolved around creating a chatbot that acts as an AI-powered travel companion. The aim was to enhance the experience of planning a trip to the UK, making it quicker, smarter, and more tailored to individual needs. Through engaging conversations, the chatbot utilises Natural Language Processing (NLP) and real-time APIs to provide users with personalised suggestions for dining, places to stay, transportation options, and attractions to explore.

As I worked on the project, I concentrated on making the experience easy to navigate for non-technical users, while also building a powerful and scalable backend that could handle dynamic requests seamlessly. Key features include:

- Custom NER (Named Entity Recognition) to extract location, food, and activity preferences
- Synonym matching and intent detection to better understand user goals
- Real-time data integration via APIs for up-to-date local recommendations
- Stateful trip planning that builds an itinerary over multiple chat turns

I ran into a few bumps while training and rolling out a tailored spaCy NER model, particularly with its size and integration issues. However, I was able to overcome these challenges by creating a more streamlined version that still offered strong entity recognition. This process of learning and adjustment significantly improved the system's performance and dependability.

In the long run, the chatbot evolved from a simple tech development into a great illustration of how artificial intelligence can help with everyday tasks, like planning a trip, making them easier, more relatable, and even a bit of fun.





