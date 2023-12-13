# Welcome to XSC Learning Notes

The static website is used for personal learning notes.

# Some Markdown Skill

## code block and inline code {#code-block-and-inline-code data-toc-label="code block and inline code"}

``` python
print("Hello World")
```

This is `inline` code.

## link and footnotes

Hyperlink used to switch to a [`correct website`][correct website].

We can also link to [`somewhere`][somewhere] of the context.

Setting footnotes[^1] is similar to the operation of link. 

[correct website]: https://www.bing.com
[somewhere]: #code-block-and-inline-code
[^1]: Here can add some footnotes.

## group content

Text and code can be grouped.

=== "code"

    ``` c
    printf("Hello World");
    ```

=== "text"

    * STEP1: ...
    * STEP2: ...
    * STEP3: ...

## table

Create a table as following.

| Object      | Description                          |
| ----------- | ------------------------------------ |
| A           | ...                                  |
| B           | ...                                  |
| C           | ...                                  |

## grid

Create grids as following.

<div class="grid cards" markdown>

* Advantage1
* Advantage2
* Advantage3
* Advantage4

</div>

## image

Image cam be added by HTML.

<figure markdown>
  ![Image title](https://dummyimage.com/600x400/){ width="300" }
  <figcaption>Image caption</figcaption>
</figure>



