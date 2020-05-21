# Git Cheat Sheet

- Change commit owner name

Here, you want to change current_name and current_email with new_name and new_email. Run the following command in one line. 
For clarity, it is split in multiple lines.

```Git
git filter-branch -f --env-filter "GIT_AUTHOR_NAME='new_name';
GIT_AUTHOR_EMAIL='new_email@example.com'; 
GIT_COMMITTER_NAME='current_name'; 
GIT_COMMITTER_EMAIL='cur_email@example.com';" 
HEAD;
```

Then, run:

```git
git push --force origin master
```


