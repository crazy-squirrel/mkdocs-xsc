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

- Inroduction
- Relative Work
- Experiment
- Conclusion

</div>

## image

Image cam be added by HTML.

=== "From Web"

    <figure markdown>
      ![Image title](https://dummyimage.com/600x400/){ width="300" }
      <figcaption>From Web</figcaption>
    </figure>

=== "From Local"

    <figure markdown>
      ![Image title](./Image/squirrel.jpg){ width="300" }
      <figcaption>From Local</figcaption>
    </figure>

## math

Syntax block as Following.

$$
\int^{\pi}_0 sinx dx = 2
$$

Inline syntax as $sinx = \lim_{n \to \infty}\sum^n_0 (-1)^{n}\frac{1}{(2n+1)!} x^{2n+1}$

## emoji

Search the emoji from the [`database`][database].

:smiley: :sweat_smile: :smiling_face_with_3_hearts: :fox: :man_mage:

[database]: https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/

# To Be Added