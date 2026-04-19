# Tool Schema Documentation

This document provides a clear description of each tool/function the agent can call, its parameters, and return types. All tools implicitly receive a `user_id` injected by the agent's state or provided via MCP to ensure multi-user isolation.

## `add_note`
Create a new note and return it.
- **Parameters:**
  - `title` (str): The title of the note.
  - `body` (str): The main content of the note.
  - `tags` (Optional[List[str]]): A list of tags for the note.
- **Returns:** 
  - `dict`: The created note represented as a dictionary.

## `get_note_by_id`
Get a specific single note by its exact database ID.
- **Parameters:**
  - `id` (str): The exact database ID of the note to retrieve.
- **Returns:** 
  - `dict`: The retrieved note represented as a dictionary.
- **Raises:** 
  - `ValueError` if the note with the given ID is not found.

## `search_notes`
Search notes by exact text query, tags, and date range.
- **Parameters:**
  - `query` (str): An exact phrase or keyword to search for in note titles or bodies. Default is `""`.
  - `tags` (Optional[List[str]]): A list of specific tags to filter by. MUST be an array of strings.
  - `date` (Optional[datetime]): Start date to filter notes created on or after this time.
  - `date_end` (Optional[datetime]): End date to filter notes created on or before this time.
  - `limit` (int): Maximum number of notes to return. Default is `10`.
- **Returns:** 
  - `List[dict]`: A list of dictionaries representing the matching notes.

## `semantic_search`
Search notes based on semantic meaning, concepts, or natural language questions using the vector database.
- **Parameters:**
  - `query` (str): A natural language sentence or concept to search for in your notes.
  - `limit` (int): Maximum number of semantic results to return. Default is `5`.
- **Returns:** 
  - `List[dict]`: A list of dictionaries representing the semantically matching notes.

## `update_note`
Update an existing note's title, body, or tags. You MUST provide the correct note ID first.
- **Parameters:**
  - `id` (str): The exact database ID of the note to update.
  - `title` (Optional[str]): The new title for the note.
  - `body` (Optional[str]): The new content body for the note.
  - `tags` (Optional[List[str]]): A new list of tags.
- **Returns:** 
  - `dict`: The updated note represented as a dictionary.
- **Raises:** 
  - `ValueError` if the note with the given ID is not found.

## `delete_note`
Delete a note completely using its exact database ID from both relational and vector databases.
- **Parameters:**
  - `id` (str): The exact database ID of the note to delete.
- **Returns:** 
  - `bool`: `True` if the deletion was successful.
- **Raises:** 
  - `ValueError` if the note with the given ID is not found.
