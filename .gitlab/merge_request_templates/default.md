<!-- Add the MR commit message here -->

### Mandatory steps

* [ ] No new warnings in build
* [ ] Every header has include guards matching the path of the file
* [ ] Documentation added to API functions/types
* [ ] Every file has a copyright declaration
* [ ] Code adheres to naming conventions
* [ ] Tests added for new features
* [ ] Make branch mergeable by other users (enables rebase + automerge)

##### CMake checks:

* [ ] New files added to CMake
* [ ] CMake files use the helper functions where appropriate
* [ ] Objects added to core libraries and tests

##### Code checks:

* [ ] Every class with a destructor has a (possibly default or deleted) copy/move constructor
* [ ] Every class with a virtual function has a virtual destructor
* [ ] References used where possible to avoid unnecessary copies

