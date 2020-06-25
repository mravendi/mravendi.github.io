# Git Cheat Sheet

- view all of your git settings 
```Git
$ git config --list --show-origin
```

- view username settings
```
$ git config user.name
```

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


- show remote url
```
$ git remote -v
```
this will list the remote urls. You can have more than one remote urls.

- remove a remote url
```
$ git remote rm <remote_name>
```

For example, if you have three remote urls:
```
$ git remote rm origin
$ git remote rm gh
$ git remote rm bb
```


- set a new remote url
```
$ git remote add <remote-name> <new-remote-url>
```



- process to push to two remote urls

assume, currently there is one origin and two remote urls named url_1 and url_2
```
$ git remote add origin <url_1>
$ git remote add <name_1> <url_1>
$ git remote add <name_2> <url_2>
```

To set up the push URLs:
```
$git remote set-url --add --push origin <url_1>
$ git remote set-url --add --push origin <url_2>
```


verify
```
$ git remote show origin
```







