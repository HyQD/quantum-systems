#!/bin/bash

user_name='Travis CI'
user_email='travis@travis-ci.org'

if [ -n "$(git status --porcelain)" ]; then
    git add .
    git -c user.name="$user_name" -c user.email="$user_email" commit -m "Black files"
    git push https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG HEAD:master --quiet
fi
