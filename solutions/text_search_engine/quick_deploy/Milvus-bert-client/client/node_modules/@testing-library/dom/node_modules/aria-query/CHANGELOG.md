# aria-query Change Log

## 1.0.0

- Updated values of aria-haspopup to include ARIA 1.1 role values
- Added the CHANGELOG file

## 2.0.0

- Remove package-lock file.
- Add Watchman config file.

## 2.0.1

- Added aria-errormessage to the ARIA Props Map.

## 3.0.0

- Bumping to a major version because of a previous breaking change.

## 4.0.0

- 912e515 (origin/fix-travis, fix-travis) Move allowed failures to excludes in Travis. The current failures are simply version incompatibilities.
- 17f4203 (origin/fixe-all-roles-html-mappings, fixe-all-roles-html-mappings) Fix all inherent ARIA role to HTML mappings
- 4ce2a9e (origin/fix-textbox, fix-textbox) Fix HTML relatedConcepts for textbox and combobox
- 8cbdf1d (origin/fix-select-mapping, fix-select-mapping) Remove baseConcepts as a prop.
- c3c510d Fix mapping for the HTML select element
- 52f2535 (origin/deprecate-requireContextRole, deprecate-requireContextRole) Deprecate the mispelled requireContextRole. Replace with requiredContextRole.
- fff3783 (origin/kurosawa-takeshi-add-double-check-tests, kurosawa-takeshi-add-double-check-tests) Update package lock file
- b90a99b (origin/kurosawa-takeshi-update-dpub-aria, kurosawa-takeshi-update-dpub-aria) Update breakUpAriaJSON script to include MapOfRoleDefinitions type on roll-up role classes-takeshi-update-dpub-aria
- 59c3199 (origin/eps1lon-fix/ie11, eps1lon-fix/ie11) Undo the eslintrc changes
- 3152480 (origin/dependabot/npm_and_yarn/eslint-6.6.0, dependabot/npm_and_yarn/eslint-6.6.0) Fix duplicate peer dependencies
- 8a661f2 Updating allowed failures for odd versions of node
- 0c85fd6 Update Travis and eslint peer dependencies
- 99df7da Bump eslint from 3.19.0 to 6.6.0

## 4.0.1

- Fix the incorrect ARIA designation of the region role to an HTML designation

## 4.0.2

- a3e2f1e Added the Copyright year (2020) for A11yance
- 3173a07 Remove Peer Dependency to ESLint

