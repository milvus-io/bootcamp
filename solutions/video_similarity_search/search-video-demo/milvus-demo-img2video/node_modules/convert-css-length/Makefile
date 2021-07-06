BIN = ./node_modules/.bin

release-patch:
	@$(call release,patch)

release-minor:
	@$(call release,minor)

release-major:
	@$(call release,major)

build:
	@$(BIN)/coffee -cb -o dist src/*

publish:
	git push --tags origin HEAD:master
	@$(BIN)/coffee -cb -o dist src/*
	npm publish

define release
	npm version $(1)
endef
