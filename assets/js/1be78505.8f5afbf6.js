(window.webpackJsonp = window.webpackJsonp || []).push([[7], {
    175: function(e, t, a) {
        "use strict";
        a.r(t);
        a(201);
        var n = a(0)
          , r = a.n(n)
          , l = a(177)
          , c = a(181)
          , i = a(58)
          , o = a(190)
          , s = a(182)
          , u = a.n(s)
          , m = a(185)
          , d = a(127)
          , f = a.n(d)
          , p = 24;
        function h(e) {
            var t = e.item
              , a = e.onItemClick
              , l = e.collapsible
              , c = t.items
              , i = t.href
              , o = t.label
              , s = t.type
              , d = Object(n.useState)(t.collapsed)
              , f = d[0]
              , p = d[1]
              , b = Object(n.useState)(null)
              , v = b[0]
              , E = b[1];
            t.collapsed !== v && (E(t.collapsed),
            p(t.collapsed));
            var k = Object(n.useCallback)((function(e) {
                e.preventDefault(),
                p((function(e) {
                    return !e
                }
                ))
            }
            ));
            switch (s) {
            case "category":
                return c.length > 0 && r.a.createElement("li", {
                    className: u()("menu__list-item", {
                        "menu__list-item--collapsed": f
                    }),
                    key: o
                }, r.a.createElement("a", {
                    className: u()("menu__link", {
                        "menu__link--sublist": l,
                        "menu__link--active": l && !t.collapsed
                    }),
                    href: "#!",
                    onClick: l ? k : void 0
                }, o), r.a.createElement("ul", {
                    className: "menu__list"
                }, c.map((function(e) {
                    return r.a.createElement(h, {
                        key: e.label,
                        item: e,
                        onItemClick: a,
                        collapsible: l
                    })
                }
                ))));
            case "link":
            default:
                return r.a.createElement("li", {
                    className: "menu__list-item",
                    key: o
                }, r.a.createElement(m.a, {
                    activeClassName: "menu__link--active",
                    className: "menu__link",
                    exact: !0,
                    to: i,
                    onClick: a
                }, o))
            }
        }
        var b = function(e) {
            var t = Object(n.useState)(!1)
              , a = t[0]
              , l = t[1]
              , c = e.docsSidebars
              , i = e.location
              , o = e.sidebar
              , s = e.sidebarCollapsible;
            if (!o)
                return null;
            var m = c[o];
            if (!m)
                throw new Error('Cannot find the sidebar "' + o + '" in the sidebar config!');
            return s && m.forEach((function(e) {
                return function e(t, a) {
                    var n = t.items
                      , r = t.href;
                    switch (t.type) {
                    case "category":
                        var l = n.map((function(t) {
                            return e(t, a)
                        }
                        )).filter((function(e) {
                            return e
                        }
                        )).length > 0;
                        return t.collapsed = !l,
                        l;
                    case "link":
                    default:
                        return r === a.pathname.replace(/\/$/, "")
                    }
                }(e, i)
            }
            )),
            r.a.createElement("div", {
                className: f.a.sidebar
            }, r.a.createElement("div", {
                className: u()("menu", "menu--responsive", {
                    "menu--show": a
                })
            }, r.a.createElement("button", {
                "aria-label": a ? "Close Menu" : "Open Menu",
                className: "button button--secondary button--sm menu__button",
                type: "button",
                onClick: function() {
                    l(!a)
                }
            }, a ? r.a.createElement("span", {
                className: u()(f.a.sidebarMenuIcon, f.a.sidebarMenuCloseIcon)
            }, "\xd7") : r.a.createElement("svg", {
                className: f.a.sidebarMenuIcon,
                xmlns: "http://www.w3.org/2000/svg",
                height: p,
                width: p,
                viewBox: "0 0 32 32",
                role: "img",
                focusable: "false"
            }, r.a.createElement("title", null, "Menu"), r.a.createElement("path", {
                stroke: "currentColor",
                strokeLinecap: "round",
                strokeMiterlimit: "10",
                strokeWidth: "2",
                d: "M4 7h22M4 15h22M4 23h22"
            }))), r.a.createElement("ul", {
                className: "menu__list"
            }, m.map((function(e) {
                return r.a.createElement(h, {
                    key: e.label,
                    item: e,
                    onItemClick: function() {
                        l(!1)
                    },
                    collapsible: s
                })
            }
            )))))
        }
          , v = a(1)
          , E = a(219)
          , k = a(9)
          , g = (a(129),
        function(e) {
            return function(t) {
                var a = t.id
                  , n = Object(k.a)(t, ["id"]);
                return a ? r.a.createElement(e, n, r.a.createElement("a", {
                    "aria-hidden": "true",
                    tabIndex: "-1",
                    className: "anchor",
                    id: a
                }), r.a.createElement("a", {
                    "aria-hidden": "true",
                    tabIndex: "-1",
                    className: "hash-link",
                    href: "#" + a,
                    title: "Direct link to heading"
                }, "#"), n.children) : r.a.createElement(e, n)
            }
        }
        )
          , y = a(130)
          , w = a.n(y)
          , N = {
            code: function(e) {
                var t = e.children;
                return "string" == typeof t ? r.a.createElement(E.a, e) : t
            },
            a: function(e) {
                return /\.[^./]+$/.test(e.href) ? r.a.createElement("a", e) : r.a.createElement(m.a, e)
            },
            pre: function(e) {
                return r.a.createElement("pre", Object(v.a)({
                    className: w.a.mdxCodeBlock
                }, e))
            },
            h1: g("h1"),
            h2: g("h2"),
            h3: g("h3"),
            h4: g("h4"),
            h5: g("h5"),
            h6: g("h6")
        }
          , C = a(204)
          , _ = a(191)
          , O = a(131)
          , j = a.n(O);
        t.default = function(e) {
            var t, a, n = e.route, s = e.docsMetadata, u = e.location, m = s.permalinkToSidebar, d = s.docsSidebars, f = s.version, p = m[u.pathname.replace(/\/$/, "")], h = Object(c.a)().siteConfig, v = (h = void 0 === h ? {} : h).themeConfig, E = (void 0 === v ? {} : v).sidebarCollapsible, k = void 0 === E || E;
            return t = n.routes,
            a = u.pathname,
            t.some((function(e) {
                return Object(_.a)(a, e)
            }
            )) ? r.a.createElement(o.a, {
                version: f
            }, r.a.createElement("div", {
                className: j.a.docPage
            }, p && r.a.createElement("div", {
                className: j.a.docSidebarContainer
            }, r.a.createElement(b, {
                docsSidebars: d,
                location: u,
                sidebar: p,
                sidebarCollapsible: k
            })), r.a.createElement("main", {
                className: j.a.docMainContainer
            }, r.a.createElement(l.a, {
                components: N
            }, Object(i.a)(n.routes))))) : r.a.createElement(C.default, e)
        }
    },
    177: function(e, t, a) {
        "use strict";
        a.d(t, "a", (function() {
            return i
        }
        )),
        a.d(t, "b", (function() {
            return m
        }
        ));
        var n = a(0)
          , r = a.n(n)
          , l = r.a.createContext({})
          , c = function(e) {
            var t = r.a.useContext(l)
              , a = t;
            return e && (a = "function" == typeof e ? e(t) : Object.assign({}, t, e)),
            a
        }
          , i = function(e) {
            var t = c(e.components);
            return r.a.createElement(l.Provider, {
                value: t
            }, e.children)
        };
        var o = "mdxType"
          , s = {
            inlineCode: "code",
            wrapper: function(e) {
                var t = e.children;
                return r.a.createElement(r.a.Fragment, {}, t)
            }
        }
          , u = Object(n.forwardRef)((function(e, t) {
            var a = e.components
              , n = e.mdxType
              , l = e.originalType
              , i = e.parentName
              , o = function(e, t) {
                var a = {};
                for (var n in e)
                    Object.prototype.hasOwnProperty.call(e, n) && -1 === t.indexOf(n) && (a[n] = e[n]);
                return a
            }(e, ["components", "mdxType", "originalType", "parentName"])
              , u = c(a)
              , m = n
              , d = u[i + "." + m] || u[m] || s[m] || l;
            return a ? r.a.createElement(d, Object.assign({}, {
                ref: t
            }, o, {
                components: a
            })) : r.a.createElement(d, Object.assign({}, {
                ref: t
            }, o))
        }
        ));
        function m(e, t) {
            var a = arguments
              , n = t && t.mdxType;
            if ("string" == typeof e || n) {
                var l = a.length
                  , c = new Array(l);
                c[0] = u;
                var i = {};
                for (var s in t)
                    hasOwnProperty.call(t, s) && (i[s] = t[s]);
                i.originalType = e,
                i[o] = "string" == typeof e ? e : n,
                c[1] = i;
                for (var m = 2; m < l; m++)
                    c[m] = a[m];
                return r.a.createElement.apply(null, c)
            }
            return r.a.createElement.apply(null, a)
        }
        u.displayName = "MDXCreateElement"
    },
    204: function(e, t, a) {
        "use strict";
        a.r(t);
        var n = a(0)
          , r = a.n(n)
          , l = a(190);
        t.default = function() {
            return r.a.createElement(l.a, {
                title: "Page Not Found"
            }, r.a.createElement("div", {
                className: "container margin-vert--xl"
            }, r.a.createElement("div", {
                className: "row"
            }, r.a.createElement("div", {
                className: "col col--6 col--offset-3"
            }, r.a.createElement("h1", {
                className: "hero__title"
            }, "Page Not Found"), r.a.createElement("p", null, "We could not find what you were looking for."), r.a.createElement("p", null, "Please contact the owner of the site that linked you to the original URL and let them know their link is broken.")))))
        }
    }
}]);
