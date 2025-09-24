# General rules
- Be sure to use the tools available to you as described in @/.roo/rules/rules.md
- Under no circumstances should you simplify or mock away real parts of the pipeline for integration testing. 
## Anti-patterns:
  - Never have more than one implementation of the same (or substantially similar) thing, 
  - Never call something by more than one name across the project unless it is absolutely necessary,
  - Never have more than one "sole source of truth" for any value.

# Testing rules
- **how to run tests** Use `python -m pytest` followed by normal arguments. Do not generate html coverage reports.

# MCP Server Tool Usage Rules

These are self-imposed rules for using the configured MCP (Model Context Protocol) server tools effectively and appropriately in my operations as Roo, the software engineer AI.

## General Principles
- **Prioritize MCP tools** Always
- **Use MCP tools first** for their designated purposes before falling back to general tools.
- **Combine MCP and general tools** when needed for comprehensive task completion.
- **Respect server limitations** such as allowed directories for filesystem operations.

## Specific Tool Usage Rules

### 1. context7 Server (Library Documentation)
- **When to use**: Always use when needing up-to-date documentation, code examples, or API references for any library, framework, or package.
- **Available tools**:
  - `resolve-library-id`: Resolves a package/product name to a Context7-compatible library ID.
  - `get-library-docs`: Fetches up-to-date documentation for a library using a Context7-compatible library ID.
- **Required workflow**: First call `resolve-library-id` to get the correct Context7-compatible library ID, then use `get-library-docs` with that ID.
- **Do not use for**: General web searches (use kagi instead).
- **Example scenarios**: Getting PyTorch documentation, finding examples for specific libraries, understanding API changes.

### 2. kagi Server (Web Search and Summarization)
- **When to use**: For general web searches, fetching information from websites, or summarizing web content.
- **Preferred tools**:
  - `kagi_search_fetch`: For keyword-based searches across multiple queries.
  - `kagi_summarizer`: For summarizing specific URLs or documents.
- **Do not use for**: Library-specific documentation (use context7 instead).
- **Example scenarios**: Researching new technologies, finding tutorials, getting current information on topics.

### 3. filesystem Server (File Operations)
- **When to use**: For file operations.
- **Preferred over general tools when**: Always.
- **Available tools**:
  - `read_file`: Read the complete contents of a file as text (deprecated, use read_text_file instead).
  - `read_text_file`: Read the complete contents of a file from the file system as text with encoding support.
  - `read_media_file`: Read an image or audio file, returns base64 encoded data and MIME type.
  - `read_multiple_files`: Read the contents of multiple files simultaneously.
  - `write_file`: Create a new file or completely overwrite an existing file with new content.
  - `edit_file`: Make line-based edits to a text file with git-style diff preview.
  - `create_directory`: Create a new directory or ensure a directory exists.
  - `list_directory`: Get a detailed listing of all files and directories in a specified path.
  - `list_directory_with_sizes`: Get a detailed listing including file sizes.
  - `directory_tree`: Get a recursive tree view of files and directories as a JSON structure.
  - `move_file`: Move or rename files and directories.
  - `search_files`: Recursively search for files and directories matching a pattern.
  - `get_file_info`: Retrieve detailed metadata about a file or directory.
- **Fallback to general tools**: When filesystem MCP is not available or returns an error.

### 4. sequentialthinking Server (Problem-Solving)
- **When to use**: For complex, multi-step problems that require dynamic thinking, revision of approaches, or maintaining context over multiple steps.
- **Available tools**:
  - `sequentialthinking`: A detailed tool for dynamic and reflective problem-solving through thoughts.
- **Required workflow**: Use the `sequentialthinking` tool to break down problems, generate hypotheses, verify solutions, and iterate until satisfied.
- **Do not use for**: Simple, straightforward tasks that can be solved with direct tool usage.
- **Example scenarios**: Debugging complex issues, designing system architecture, planning multi-step implementations.


### 5. memory Server (Knowledge Graph)
- **When to use**: For managing persistent knowledge, relationships between concepts, or tracking information across conversations.
- **Available tools**:
  - `create_entities`: Create multiple new entities in the knowledge graph.
  - `create_relations`: Create multiple new relations between entities (relations should be in active voice).
  - `add_observations`: Add new observations to existing entities.
  - `delete_entities`: Delete multiple entities and their associated relations.
  - `delete_observations`: Delete specific observations from entities.
  - `delete_relations`: Delete multiple relations from the knowledge graph.
  - `read_graph`: Read the entire knowledge graph.
  - `search_nodes`: Search for nodes based on a query.
  - `open_nodes`: Open specific nodes in the knowledge graph by their names.
- **Maintenance**: Regularly review and update the knowledge graph to maintain accuracy.
- **Example scenarios**: Tracking project dependencies, remembering user preferences, maintaining context across sessions.

### 6. pycharm Server (IDE Operations)
- **When to use**: When working with python code. Especially when discovering project structure, performing refactoring, and for code analysis.
- **Available tools**:
  - `get_file_problems`: Analyzes the specified file for errors and warnings using IntelliJ's inspections.
  - `get_project_dependencies`: Get a list of all dependencies defined in the project.
  - `get_project_modules`: Get a list of all modules in the project with their types.
  - `get_project_problems`: Retrieves all project problems (errors, warnings, etc.) detected in the project.
  - `find_files_by_glob`: Searches for all files whose relative paths match the specified glob pattern.
  - `find_files_by_name_keyword`: Searches for all files whose names contain the specified keyword.
  - `list_directory_tree`: Provides a tree representation of the specified directory.
  - `reformat_file`: Reformats a specified file in the JetBrains IDE.
  - `replace_text_in_file`: Replaces text in a file with flexible options for find and replace operations.
  - `search_in_files_by_regex`: Searches with a regex pattern within all files in the project.
  - `search_in_files_by_text`: Searches for a text substring within all files in the project.
  - `get_symbol_info`: Retrieves information about the symbol at the specified position in the specified file.
  - `rename_refactoring`: Renames a symbol (variable, function, class, etc.) in the specified file.
  - `find_commit_by_message`: Searches for a commit based on the provided text or keywords in the project history.
  - `get_project_vcs_status`: Retrieves the current version control status of files in the project.
- **Do not use for**: General file operations (use filesystem MCP or general tools).
- **Example scenarios**: Running tests, checking for code issues, performing refactoring operations.

### 7. arXivPaper Server (Academic Papers)
- **When to use**: When writing documentation that covers academic topics; for searching, retrieving, or analyzing academic papers from arXiv, for writing citations and references.
- **Available tools**:
  - `scrape_recent_category_papers`: Crawls recent pages for specific categories to get latest papers.
  - `search_papers`: Searches for papers by keyword.
  - `get_paper_info`: Retrieves detailed information for a specific paper by ID.
  - `analyze_trends`: Analyzes trends in a specific category over a given time period.
- **Do not use for**: General web searches (use kagi instead).
- **Example scenarios**: Researching state-of-the-art methods, finding related work, staying updated on academic developments.

## Integration Rules
- **Tool chaining**: Use MCP tools in sequence when one tool's output informs another's input (e.g., search with kagi, then summarize with sequentialthinking).
- **Fallback strategy**: If an MCP tool fails or is unavailable, gracefully fall back to general tools or ask for clarification.
- **Efficiency**: Choose the most direct MCP tool for the task to minimize tool usage and response time.
- **Context awareness**: Consider the current mode and project context when selecting MCP tools.

## Maintenance Rules
- **Review periodically**: Update these rules as new MCP servers are added or existing ones are modified.
- **Test usage**: Verify that MCP tools work as expected in different scenarios.
- **Document exceptions**: Note any cases where general tools are preferred over MCP tools for specific reasons.