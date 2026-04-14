TaskList API Reference — Chainlit (concise developer-focused summary)

What TaskList does
- TaskList is a UI element that can be sent directly to the chat interface (not attached to a Message or Step). It provides a lightweight task-tracking surface inside the chat, allowing you to display and update a list of tasks as part of a chat workflow.

Key types and surface
- TaskList: container for tasks. Properties include status (string) that reflects overall list state. Methods include add_task(Task) and send() to render updates to the UI.
- Task: individual task item with at least a title (string) and a status (TaskStatus). Optional: forId (to link to a chat Message for navigation from the task).
- TaskStatus: enum-like values (as seen in examples) such as RUNNING, DONE, FAILED. These are used to describe a task's current state.

Core props/attributes (as evidenced by API usage)
- TaskList.status: string; example sets it to "Running..." or "Failed".
- Task.title: string; e.g. "Processing data".
- Task.status: TaskStatus; e.g. TaskStatus.RUNNING.
- Task.forId: message identifier to link a Task to a Message for navigation.

Key methods/events
- TaskList.add_task(task): async; adds a Task to the list.
- TaskList.send(): renders the current TaskList state to the chat UI.
- TaskList.status: readable string reflecting overall status; can be updated to reflect aggregate state.
- Task properties can be updated (e.g., task1.title, task1.status) before calling send() again.

Usage example (from API page)

import chainlit as cl

@cl.on_chat_start
async def main():
  # Create the TaskList
  task_list = cl.TaskList()
  task_list.status = "Running..."

  # Create a task and put it in the running state
  task1 = cl.Task(title="Processing data", status=cl.TaskStatus.RUNNING)
  await task_list.add_task(task1)

  # Create another task that is in the ready state
  task2 = cl.Task(title="Performing calculations")
  await task_list.add_task(task2)

  # Optional: link a message to each task to allow task navigation in the chat history
  message = await cl.Message(content="Started processing data").send()
  task1.forId = message.id

  # Update the task list in the interface
  await task_list.send()

  # Perform some action on your end
  await cl.sleep(1)

  # Update the task statuses
  task1.status = cl.TaskStatus.DONE
  task2.status = cl.TaskStatus.FAILED
  task_list.status = "Failed"
  await task_list.send()

Was this page helpful?

Typical usage scenarios
- Show progress of a multi-step data processing pipeline directly in chat.
- Coordinate background tasks with visible statuses for users guiding a conversation.
- Link task items to specific chat messages for quick navigation from a task to its context.

Source
- TaskList - Chainlit API Reference: https://docs.chainlit.io/api-reference/elements/tasklist
