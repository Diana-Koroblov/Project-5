Guidelines for Writing Professional Software at the Highest Level of Excellence
The Professional Programmer in the Age of AI

A professional software engineer is not just someone who can write code, but someone who understands the full software lifecycle: planning, documenting, testing, maintaining, and improving complex systems. Professional work means high standards of quality, consistency, clear communication, teamwork, and responsibility.

Core characteristics of a professional programmer include:

• Systems thinking - understanding how each component fits into the overall system.
• Planning before implementation - writing requirements, architecture, and planning documents before coding.
• Uncompromising quality - producing clean, documented, tested, and secure code.
• Continuous learning - staying current with technologies, tools, and methods.
• Effective communication - explaining technical solutions clearly to different audiences.

Software development is usually a team effort. Typical teams include architects, developers, QA engineers, product managers, and sometimes DevOps engineers. Professional teams use structured workflows such as Scrum, Kanban, or SAFe, with daily standups, sprint reviews, retrospectives, code reviews, and shared coding standards.

Every professional project should follow a defined Software Development Life Cycle (SDLC):

Requirements definition
Planning and architecture
Development
Testing
Deployment
Maintenance and improvement

In the AI era, prompt-driven development or Vibe Coding changes how software is built. A programmer can orchestrate AI agents to generate, review, and improve code, effectively functioning like a senior architect managing multiple assistants in parallel. But this only works well if requirements, planning, and documentation are defined clearly in advance. Without that, AI may generate code that works superficially but does not meet professional standards.

The first rule of professional AI-assisted coding is therefore: define and require complete documentation before writing code.

Mandatory Project Structure and Documentation

Every professional project must include a minimum folder structure and mandatory documentation. Without them, the project does not meet the minimum standard.

README.md in the project root

Every project must include a README.md file in the root directory. It should function as a full user manual and include:

• Installation instructions
• Usage instructions
• Examples and demonstrations
• Configuration guide
• Contribution guidelines
• License and third-party credits

Mandatory docs/ folder

Every project must include a docs/ folder containing at least:

• docs/PRD.md - Product Requirements Document  
  This should define the purpose of the project, user problem, target audience, goals, KPIs, acceptance criteria, user stories, functional and non-functional requirements, assumptions, dependencies, constraints, out-of-scope items, and milestones.

• docs/PLAN.md - Planning and architecture document  
  This should describe the technical design, architecture diagrams, deployment diagrams, UML where needed, architectural decisions and trade-offs, APIs, interfaces, data schemas, and contracts.

• docs/TODO.md - Task tracking document  
  This should list all required tasks with priorities, statuses, phases, ownership, milestones, and definition of done.

Separate PRDs for major mechanisms

Every important algorithm, mechanism, or major technical component must have its own dedicated PRD document, such as:

• docs/PRDmlalgorithm.md
• docs/PRDauthentication.md
• docs/PRDsearchengine.md
• docs/PRDcaching.md

Each such PRD should include:

• Detailed description of the mechanism or algorithm
• Relevant theoretical background
• Inputs, outputs, constraints, and performance expectations
• Alternatives considered and rationale for the chosen approach
• Success criteria and testing scenarios

Recommended project structure

A recommended structure includes:

• src/ for source code
• tests/ for unit and integration tests
• docs/ for documentation
• config/ for configuration files
• data/ for input data
• results/ for outputs
• assets/ for images and resources
• notebooks/ for analysis notebooks
• README.md
• pyproject.toml
• uv.lock
• .env-example
• .gitignore

Mandatory workflow

The required workflow is:

Write docs/PRD.md
Write docs/PLAN.md
Write docs/TODO.md
Write dedicated PRDs for important algorithms/mechanisms
Approve all documents before development starts
Begin development and continuously update TODO.md
Save results, create visualizations, and update README.md

Code Documentation and Project Structure

Good documentation and organized structure are essential for professional software development. They make code understandable, maintainable, and easy for others to contribute to.

Modular project structure

Projects should be organized logically by role: source code, tests, documentation, configuration, data, results, and assets. This can be feature-based or layered, as long as code, data, outputs, and documentation remain clearly separated.

File size rule: maximum 150 lines

Each code file must not exceed 150 lines of code. Blank lines and comment lines do not count.

If a file exceeds the limit, it must be split. Code must never be compressed unnaturally just to satisfy the limit.

Recommended splitting strategies include:

• Extract helper functions
• Extract constants into constants.py
• Extract models into separate files
• Split classes or modules by responsibility
• Use mixins or base classes when logic is shared

Code quality and comments

Code quality is not measured only by functionality, but also by readability and maintainability.

Requirements include:

• Clear and descriptive names
• Short, focused functions
• Single-responsibility design
• No duplicated code
• Consistent style across the project
• Full docstrings for modules, classes, and functions
• Comments that explain why, not only what
• Documentation of assumptions, edge cases, and design choices
• Updating comments when code changes

SDK Architecture and Object-Oriented Design
SDK-based architecture

All business logic must be accessible through an SDK layer. The SDK serves as the single entry point for all consumers: GUI, CLI, third-party integrations, and future services.

This means:

• No business logic inside GUI or CLI layers
• External consumers should use the SDK, not internal modules directly
• Internal architecture should separate business logic from infrastructure such as databases, files, and external APIs

Object-oriented design without duplication

Code should follow OOP principles and avoid duplication. If the same logic appears in multiple places, it should be extracted into:

• A shared module
• A base class
• A mixin
• A wrapper function
• A suitable design pattern such as Template Method

Mixins should be small, focused, independently testable, and not override each other unexpectedly.

API Gatekeeper and Rate Control

All external API calls must pass through a central API Gatekeeper.

The gatekeeper is responsible for:

• Enforcing rate limits
• Managing queues
• Handling retries
• Monitoring and logging all API calls

Direct API calls that bypass the gatekeeper are forbidden.

Rate-limit configuration

All rate limits must be loaded from configuration files, never hardcoded into source code.

Typical configuration includes:

• Requests per minute
• Requests per hour
• Maximum concurrent requests
• Retry delay
• Maximum retries

Queue management

When rate limits are reached, requests must be queued rather than dropped or allowed to crash the system.

Requirements include:

• FIFO queue
• Configurable maximum queue depth
• Backpressure when full
• Drain mechanism when rate windows reset

Test-Driven Development and Quality Assurance

All development must follow TDD: Red -> Green -> Refactor.

Testing requirements
• Every module must have a matching test file
• Every public function or method must have at least one test
• Tests must cover both normal and failure paths
• Tests should be written before or alongside implementation, not afterward
• Shared fixtures should be placed in conftest.py
• External dependencies such as APIs, databases, and files should be mocked when appropriate
• Test files must also obey the 150-line rule
• Tests must not depend on live external services

Minimum coverage

Global test coverage must be at least 85%. The test suite should fail automatically if coverage drops below that threshold.

Coverage should include:

• Statement coverage
• Branch coverage
• Path coverage for critical flows

Edge cases and failures

Edge cases must be identified, documented, and tested. Error handling should include:

• Defensive programming
• Clear error messages
• Detailed logging
• Graceful degradation where possible

Expected test results should be documented, including pass/fail reports and logs for successful and failed runs.

Linting, Configuration Management, and Security
Zero lint violations

All code must pass linting with zero violations, specifically under Ruff.

Expected rule categories include:

• Style and formatting
• Undefined names and unused imports
• Import ordering
• Naming conventions
• Modern Python upgrades
• Common bug patterns
• Comprehension usage
• Simplification opportunities

No hardcoded configurable values

All configurable values must come from configuration files, not from source code.

This includes:

• API URLs
• Rate limits
• Timeouts
• Credentials
• Service-specific parameters

Values allowed in code include:

• Physical or mathematical constants
• Enums
• immutable constants in dedicated constants files
• Reasonable default parameter values

Configuration hierarchy

Configuration should follow a clear hierarchy with versioned config files, for example:

• config/setup.json
• config/ratelimits.json
• config/loggingconfig.json
• .env for secrets
• .env-example for placeholders
• pyproject.toml for build, lint, and test settings
• constants.py for immutable project constants

Secrets management

No secrets may be stored in the repository.

Requirements:

• Never commit API keys, tokens, passwords, or credential files
• Use environment variables only
• Include .env, .pem, .key, and credential files in .gitignore
• Include .env-example with dummy values
• In production, use a proper secrets manager
• Rotate keys periodically
• Monitor usage
• Apply least-privilege access

Version Control and Dependency Management with uv
Version tracking

Both code and configuration must include explicit version tracking. Versions should start at 1.00 and increase with significant changes.

Required version locations include:

• Code version file
• Configuration version field
• Rate-limit configuration version field

The application should validate configuration compatibility at startup.

Git best practices

Professional Git usage includes:

• Clear and meaningful commit messages
• Separate branches for new features
• Pull requests
• Code reviews
• Tagging important releases
• Clean and understandable history

Prompt log

Projects developed with AI should include a Prompt Engineering Log documenting major prompts used during development, their purpose, examples of outputs, iterative improvements, and lessons learned.

uv is mandatory

All projects must use uv as the package manager and task runner. Direct use of pip, python -m, venv, or virtualenv is forbidden.

Examples:

• uv sync instead of pip install
• uv add <pkg> instead of pip install <pkg>
• uv run python script.py instead of python script.py
• uv run pytest tests/ instead of python -m pytest
• uv lock instead of pip freeze

Additional requirements:

• pyproject.toml is the single source of truth for dependencies
• uv.lock must exist and be version-controlled
• No direct pip calls in scripts, docs, or CI/CD
• All tools must run via uv run

Research and Results Analysis

What separates ordinary software work from excellent work is the depth of research and analysis.

Parameter exploration

Projects should include systematic parameter sensitivity analysis:

• Controlled experiments
• Accurate documentation of parameter effects
• Quantitative comparison
• Methods such as one-at-a-time analysis, variance-based analysis, or partial derivatives where appropriate

Results analysis notebook

A results analysis notebook is a central research artifact. It should include:

• Methodical analysis of experiment results
• Comparison between algorithms, configurations, or approaches
• Mathematical derivations or theoretical analysis where relevant
• Equations and formulas written properly
• References to academic literature where appropriate

Visual presentation of results

High-quality visualizations are essential. Appropriate chart types may include:

• Bar charts
• Line charts
• Scatter plots
• Heatmaps
• Box plots
• Waterfall charts

Graphs should have:

• Clear labels
• Consistent and accessible colors
• Detailed captions
• Clear legends
• High resolution

User Interface and User Experience

UI and UX are critical to the success of any software system.

Quality criteria

Usability should be evaluated according to criteria such as:

• Learnability
• Efficiency
• Memorability
• Error prevention
• User satisfaction

The document also references Nielsen's heuristics, including:

• Visibility of system status
• Match between system and the real world
• User control and freedom
• Consistency and standards
• Error prevention
• Recognition rather than recall
• Flexibility and efficiency of use
• Aesthetic and minimalist design
• Helping users recover from errors
• Help and documentation

Interface documentation

The interface should be documented with:

• Screenshots of every screen and state
• Typical workflow descriptions
• Interaction and feedback explanations
• Accessibility considerations

Costs and Pricing

Understanding development and operating cost is essential for proper project planning.

Cost analysis

Projects using paid APIs should include precise token accounting:

• Input tokens
• Output tokens
• Cost per model or provider
• Total cost estimates

Optimization strategies include:

• Reducing token usage
• Batch processing
• Choosing models based on cost-effectiveness

Budget management

Projects should include:

• Cost forecasts at scale
• Real-time usage monitoring
• Alerts for budget overruns

Extensibility and Maintainability
Extension points

Projects should be designed to support future extension without changing core logic. This can be done through:

• Plugin architectures
• Clear extension interfaces
• Lifecycle hooks
• Middleware-based design
• API-first thinking

Maintainability

Maintainable code is characterized by:

• Modularity
• Separation of concerns
• Reusability
• Analyzability
• Testability
• Ease of modification

International Quality Standards

The document refers to ISO/IEC 25010, which defines major software quality characteristics.

Key quality dimensions include:

• Functional suitability
• Performance efficiency
• Compatibility
• Usability
• Reliability
• Security
• Maintainability
• Portability

These cover correctness, response times, interoperability, learnability, maturity, fault tolerance, confidentiality, modularity, installability, and replaceability.

Organizing the Project as a Package

Code should be structured as a proper package to support reuse, dependency management, installation, testing, and distribution.

Package definition

Every package should include a package definition file, preferably pyproject.toml, containing:

• Name
• Version
• Description
• Author
• License
• Dependencies

init.py

Packages should include init.py files and use them to expose public interfaces through all and version metadata where appropriate.

Relative imports

Imports should use relative paths or package-based imports, never ad hoc absolute filesystem paths. File read/write operations should also be package-aware.

Package checklist

A proper package should include:

• Valid pyproject.toml
• Proper init.py
• Clear separation of source, tests, and docs
• Relative or package-based imports only

Parallel Processing and Performance

Modern software often requires concurrency for performance.

Multiprocessing vs multithreading
• Multiprocessing is suitable for CPU-bound tasks such as numerical computation, image processing, and model training.
• Multithreading is suitable for I/O-bound tasks such as network calls, database access, and file operations.

Each should be used according to workload characteristics, not interchangeably.

Thread safety

Concurrent code must be thread-safe. Shared state should be protected, race conditions avoided, and design choices documented. Parallelism should improve performance without harming correctness or reproducibility.

Modular Design and Building Blocks

The document emphasizes building software from reusable, well-defined building blocks.

Good building blocks should have:

• A single clear responsibility
• Defined interfaces
• Low coupling
• High cohesion
• Ease of testing
• Ease of reuse and extension

The architectural goal is to compose systems from small, reliable units rather than large tangled modules.

Final Professional Checklist

A high-quality submission should satisfy all of the following:

Structure and documentation
• Root README.md
• docs/PRD.md, docs/PLAN.md, docs/TODO.md
• Dedicated PRDs for major algorithms/mechanisms
• Clear folder organization

Architecture and code
• SDK-based entry point
• OOP without duplication
• Modular structure
• No oversized files
• Clear comments and docstrings
• Consistent naming and style

Testing and quality
• TDD workflow
• Tests for every module and public function
• Normal-path and failure-path coverage
• At least 85% total coverage
• Zero lint violations

Configuration and security
• No hardcoded secrets
• No hardcoded configurable runtime values
• Versioned configuration files
• Proper .gitignore
• .env-example
• Environment-based secret loading

Research and visualization
• Parameter analysis
• Results notebook
• High-quality graphs and figures
• Evidence-based conclusions

Extensibility and standards
• Plugin/extensibility mindset
• Maintainable architecture
• Package organization
• Alignment with recognized quality standards

Main Practical Message

The central message of the document is that excellent software is not just code that works. It is software that is:

• Planned before implementation
• Fully documented
• Modular and maintainable
• Tested rigorously
• Secure by design
• Configurable and versioned
• Professionally analyzed
• Ready for collaboration, reuse, and future extension

In the AI era, these standards become even more important, not less. AI can dramatically accelerate implementation, but only disciplined engineering produces professional software.
