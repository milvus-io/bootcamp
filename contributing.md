# Contributing to Milvus Bootcamp

👍🎉 First off, thanks for taking the time to contribute! 🎉👍

The following is a set of guidelines for contributing to Milvus Bootcamp. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.



## 🔎 What should I know before I get started?

- Milvus Bootcamp design decision

  [Milvus Bootcamp](https://github.com/milvus-io/bootcamp) is designed to expose users to both the simplicity and depth of the [**Milvus**](https://milvus.io/) vector database. Discover how to run **benchmark tests** as well as build similarity search applications spanning **chatbots**, **recommendation systems**, **reverse image search**, **molecular search**, **video search**, **audio saerch**, and more.

  Each solution in Bootcamp contains a **Jupyter Notebook** and a **Docker deployable solution**, meaning anyone can run it on their local machine.

- The code structure

  For each Docker deployable solution, you can learn the code structure and then modify the code to contribute.

  ```bash
  └───server
  │   │   Dockerfile
  │   │   requirements.txt  # Python requirement library
  │   │   main.py  # File for starting the server.
  │   │
  │   └───src
  │       │   config.py  # Configuration file.
  │       │   encode.py  # Covert image/video/text/... to embeddings.
  │       │   milvus.py  # Milvus related functions such as insert/query vectors etc.
  │       │   mysql.py   # Mysql related functions such as add/delete/query IDs and object information.
  │       │   
  │       └───operations # Call methods in server to insert/query/delete data.
  │               │   insert.py
  │               │   query.py
  │               │   delete.py
  │               │   count.py
  ```

  

## 📝 How can I contribute?

### Reporting bugs

This section guides you through submitting a bug report for Bootcamp. Actually everyone can [submit a bug issue](https://github.com/milvus-io/bootcamp/issues/new/choose) with the following template and, most importantly, in English.

> **Describe the issue**
> A clear and concise description of what the issue is.
>
> **To Reproduce**
>
> - Which solution are you running? Please post the link.
> - Steps to reproduce the behavior(Docker or Source code):
> 1. Go to '...'
> 2. Click on '....'
> 3. Scroll down to '....'
> 4. See an error
>
> **Expected behavior**
> A clear and concise description of what you expected to happen.
>
> **Screenshots**
> If applicable, add screenshots to help explain your problem.
>
> **Software version (please complete the following information):**
>  - Milvus: [e.g. 2.0.0rc1]
>  - Server: [e.g. 2.0.0]
>  - Client: [e.g. 2.0.0]
>
> **Additional context**
> Add any other context about the problem here.

### Enhancement suggestion

This section guides you through [submitting an enhancement suggestion](https://github.com/milvus-io/bootcamp/issues/new/choose) for Bootcamp, including completely new features and minor improvements to existing functionality.

> **Is your feature request related to a problem? Please describe.**
> A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]
>
> **Describe the solution you'd like**
> A clear and concise description of what you want to happen.
>
> **Describe alternatives you've considered**
> A clear and concise description of any alternative solutions or features you've considered.
>
> **Additional context**
> Add any other context or screenshots about the feature request here.

### Pull requests

When you fixed a bug or finished an enhancement suggestion, you can submit a pull request to Milvus Bootcamp, and please follow these steps to check your contribution.

> - [ ] A reference to a related issue in your repository.
>
> Each PR is related to an issue, and you need to list the issue, like https://github.com/milvus-io/bootcamp/issues/600.
>
> - [ ]  A description of the changes proposed in the pull request.
>
>   A brief introduction to this PR.
>
> - [ ] Add delight to the experience when all tasks are complete 🎉



## :books: ​Styleguides

Before submitting your contribution to Bootcamp, please note these styleguides.

- Source code in [Python](https://www.python.org/dev/peps/pep-0008/)
- Notebook in [Jupyter](https://jupyter.readthedocs.io/en/latest/contributing/ipython-dev-guide/coding_style.html)
- Documents written in [Markdown](https://www.markdownguide.org/basic-syntax/)



## 💻 Additional notes

### Issue and Pull Request Labels

This section lists the labels we use to help us track and manage issues and pull requests. The labels are loosely grouped by their purpose, but it's not required that every issue has a label from every group or that an issue can't have more than one label from the same group.

#### Type of Issue and Issue State

| Label name    | `milvus-io/bootcamp` 🔎                                       | Description                                                  |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `enhancement` | [search](https://github.com/search?q=is%3Aopen+is%3Aissue+repo%3Amilvus-io%2Fbootcamp+label%3Aenhancement) | Feature requests.                                            |
| `bug`         | [search](https://github.com/search?q=is%3Aopen+is%3Aissue+repo%3Amilvus-io%2Fbootcamp+label%3Abug) | Confirmed bugs or reports that are very likely to be bugs.   |
| `question`    | [search](https://github.com/search?q=is%3Aopen+is%3Aissue+repo%3Amilvus-io%2Fbootcamp+label%3Aquestion) | Questions more than bug reports or feature requests.         |
| `help-wanted` | [search](https://github.com/search?q=is%3Aopen+is%3Aissue+repo%3Amilvus-io%2Fbootcamp+label%3Ahelp-wanted) | Ask for help in the community.                               |
| `duplicate`   | [search](https://github.com/search?q=is%3Aopen+is%3Aissue+repo%3Amilvus-io%2Fbootcamp+label%3Aduplicate) | Issues which are duplicates of other issues, i.e. they have been reported before. |
| `wontfix`     | [search](https://github.com/search?q=is%3Aopen+is%3Aissue+repo%3Amilvus-io%2Fbootcamp+label%3Awontfix) | The community will not to fix these issues for now, either because they're working as intended or for some other reason. |
| `invalid`     | [search](https://github.com/search?q=is%3Aopen+is%3Aissue+repo%3Amilvus-io%2Fbootcamp+label%3Ainvalid) | Issues which aren't valid (e.g. user errors).                |

#### Pull Request Labels

| Label name         | `milvus-io/bootcamp` 🔎                                       | Description                                                  |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `needs-review`     | [search](https://github.com/search?q=is%3Aopen+is%3Apr+repo%3Amilvus-io%2Fbootcamp+label%3Aneeds-review) | Pull requests which need code review, and approval from maintainers or Atom core team. |
| `under-review`     | [search](https://github.com/search?q=is%3Apr+repo%3Amilvus-io%2Fbootcamp+label%3Aunder-review) | Pull requests being reviewed by others.                      |
| `requires-changes` | [search](https://github.com/search?q=is%3Apr+repo%3Amilvus-io%2Fbootcamp+label%3Arequires-changes) | Pull requests which need to be updated based on review comments and then reviewed again. |
| `needs-testing`    | [search](https://github.com/search?q=is%3Apr+repo%3Amilvus-io%2Fbooctamp+label%3Aneeds-testing) | Pull requests which need manual testing.                     |
