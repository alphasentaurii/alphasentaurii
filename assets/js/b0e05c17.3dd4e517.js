/*! For license information please see b0e05c17.3dd4e517.js.LICENSE */
(window.webpackJsonp = window.webpackJsonp || []).push([[32], {
    156: function(e, t, r) {
        "use strict";
        r.r(t),
        r.d(t, "frontMatter", (function() {
            return i
        }
        )),
        r.d(t, "rightToc", (function() {
            return s
        }
        )),
        r.d(t, "metadata", (function() {
            return c
        }
        )),
        r.d(t, "default", (function() {
            return u
        }
        ));
        var n = r(1)
          , o = r(9)
          , a = (r(178),
        r(177))
          , i = {
            id: "making-a-progressive-web-app",
            title: "Making a Progressive Web App"
        }
          , s = [{
            value: "Why Opt-in?",
            id: "why-opt-in",
            children: []
        }, {
            value: "Offline-First Considerations",
            id: "offline-first-considerations",
            children: []
        }, {
            value: "Progressive Web App Metadata",
            id: "progressive-web-app-metadata",
            children: []
        }]
          , c = {
            id: "making-a-progressive-web-app",
            title: "Making a Progressive Web App",
            description: "The production build has all the tools necessary to generate a first-class",
            source: "@site/../docs/making-a-progressive-web-app.md",
            permalink: "/docs/making-a-progressive-web-app",
            editUrl: "https://github.com/facebook/create-react-app/edit/master/docusaurus/website/../docs/making-a-progressive-web-app.md",
            lastUpdatedBy: "Peet Goddard",
            lastUpdatedAt: 1576444865,
            sidebar: "docs",
            previous: {
                title: "Adding Custom Environment Variables",
                permalink: "/docs/adding-custom-environment-variables"
            },
            next: {
                title: "Creating a Production Build",
                permalink: "/docs/production-build"
            }
        }
          , l = {
            rightToc: s,
            metadata: c
        }
          , p = "wrapper";
        function u(e) {
            var t = e.components
              , r = Object(o.a)(e, ["components"]);
            return Object(a.b)(p, Object(n.a)({}, l, r, {
                components: t,
                mdxType: "MDXLayout"
            }), Object(a.b)("p", null, "The production build has all the tools necessary to generate a first-class\n", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://developers.google.com/web/progressive-web-apps/"
            }), "Progressive Web App"), ",\nbut ", Object(a.b)("strong", {
                parentName: "p"
            }, "the offline/cache-first behavior is opt-in only"), ". By default,\nthe build process will generate a service worker file, but it will not be\nregistered, so it will not take control of your production web app."), Object(a.b)("p", null, "In order to opt-in to the offline-first behavior, developers should look for the\nfollowing in their ", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://github.com/facebook/create-react-app/blob/master/packages/cra-template/template/src/index.js"
            }), Object(a.b)("inlineCode", {
                parentName: "a"
            }, "src/index.js")), " file:"), Object(a.b)("pre", null, Object(a.b)("code", Object(n.a)({
                parentName: "pre"
            }, {
                className: "language-js"
            }), "// If you want your app to work offline and load faster, you can change\n// unregister() to register() below. Note this comes with some pitfalls.\n// Learn more about service workers: https://bit.ly/CRA-PWA\nserviceWorker.unregister();\n")), Object(a.b)("p", null, "As the comment states, switching ", Object(a.b)("inlineCode", {
                parentName: "p"
            }, "serviceWorker.unregister()"), " to\n", Object(a.b)("inlineCode", {
                parentName: "p"
            }, "serviceWorker.register()"), " will opt you in to using the service worker."), Object(a.b)("h2", {
                id: "why-opt-in"
            }, "Why Opt-in?"), Object(a.b)("p", null, "Offline-first Progressive Web Apps are faster and more reliable than traditional web pages, and provide an engaging mobile experience:"), Object(a.b)("ul", null, Object(a.b)("li", {
                parentName: "ul"
            }, "All static site assets are cached so that your page loads fast on subsequent visits, regardless of network connectivity (such as 2G or 3G). Updates are downloaded in the background."), Object(a.b)("li", {
                parentName: "ul"
            }, "Your app will work regardless of network state, even if offline. This means your users will be able to use your app at 10,000 feet and on the subway."), Object(a.b)("li", {
                parentName: "ul"
            }, "On mobile devices, your app can be added directly to the user's home screen, app icon and all. This eliminates the need for the app store.")), Object(a.b)("p", null, "However, they ", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://github.com/facebook/create-react-app/issues/2398"
            }), "can make debugging deployments more challenging"), " so, starting with Create React App 2, service workers are opt-in."), Object(a.b)("p", null, "The ", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://developers.google.com/web/tools/workbox/modules/workbox-webpack-plugin"
            }), Object(a.b)("inlineCode", {
                parentName: "a"
            }, "workbox-webpack-plugin")), "\nis integrated into production configuration,\nand it will take care of generating a service worker file that will automatically\nprecache all of your local assets and keep them up to date as you deploy updates.\nThe service worker will use a ", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://developers.google.com/web/fundamentals/instant-and-offline/offline-cookbook/#cache-falling-back-to-network"
            }), "cache-first strategy"), "\nfor handling all requests for local assets, including\n", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://developers.google.com/web/fundamentals/primers/service-workers/high-performance-loading#first_what_are_navigation_requests"
            }), "navigation requests"), "\nfor your HTML, ensuring that your web app is consistently fast, even on a slow\nor unreliable network."), Object(a.b)("h2", {
                id: "offline-first-considerations"
            }, "Offline-First Considerations"), Object(a.b)("p", null, "If you do decide to opt-in to service worker registration, please take the\nfollowing into account:"), Object(a.b)("ol", null, Object(a.b)("li", {
                parentName: "ol"
            }, Object(a.b)("p", {
                parentName: "li"
            }, "After the initial caching is done, the ", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://developers.google.com/web/fundamentals/primers/service-workers/lifecycle"
            }), "service worker lifecycle"), "\ncontrols when updated content ends up being shown to users. In order to guard against\n", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://github.com/facebook/create-react-app/issues/3613#issuecomment-353467430"
            }), "race conditions with lazy-loaded content"), ',\nthe default behavior is to conservatively keep the updated service worker in the "', Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://developers.google.com/web/fundamentals/primers/service-workers/lifecycle#waiting"
            }), "waiting"), '"\nstate. This means that users will end up seeing older content until they close (reloading is not\nenough) their existing, open tabs. See ', Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://jeffy.info/2018/10/10/sw-in-c-r-a.html"
            }), "this blog post"), "\nfor more details about this behavior.")), Object(a.b)("li", {
                parentName: "ol"
            }, Object(a.b)("p", {
                parentName: "li"
            }, "Users aren't always familiar with offline-first web apps. It can be useful to\n", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://developers.google.com/web/fundamentals/instant-and-offline/offline-ux#inform_the_user_when_the_app_is_ready_for_offline_consumption"
            }), "let the user know"), '\nwhen the service worker has finished populating your caches (showing a "This web\napp works offline!" message) and also let them know when the service worker has\nfetched the latest updates that will be available the next time they load the\npage (showing a "New content is available once existing tabs are closed." message). Showing\nthese messages is currently left as an exercise to the developer, but as a\nstarting point, you can make use of the logic included in ', Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://github.com/facebook/create-react-app/blob/master/packages/cra-template/template/src/serviceWorker.js"
            }), Object(a.b)("inlineCode", {
                parentName: "a"
            }, "src/serviceWorker.js")), ", which\ndemonstrates which service worker lifecycle events to listen for to detect each\nscenario, and which as a default, only logs appropriate messages to the\nJavaScript console.")), Object(a.b)("li", {
                parentName: "ol"
            }, Object(a.b)("p", {
                parentName: "li"
            }, "Service workers ", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://developers.google.com/web/fundamentals/getting-started/primers/service-workers#you_need_https"
            }), "require HTTPS"), ",\nalthough to facilitate local testing, that policy\n", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://stackoverflow.com/questions/34160509/options-for-testing-service-workers-via-http/34161385#34161385"
            }), "does not apply to ", Object(a.b)("inlineCode", {
                parentName: "a"
            }, "localhost")), ".\nIf your production web server does not support HTTPS, then the service worker\nregistration will fail, but the rest of your web app will remain functional.")), Object(a.b)("li", {
                parentName: "ol"
            }, Object(a.b)("p", {
                parentName: "li"
            }, "The service worker is only enabled in the ", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "/docs/deployment"
            }), "production environment"), ",\ne.g. the output of ", Object(a.b)("inlineCode", {
                parentName: "p"
            }, "npm run build"), ". It's recommended that you do not enable an\noffline-first service worker in a development environment, as it can lead to\nfrustration when previously cached assets are used and do not include the latest\nchanges you've made locally.")), Object(a.b)("li", {
                parentName: "ol"
            }, Object(a.b)("p", {
                parentName: "li"
            }, "If you ", Object(a.b)("em", {
                parentName: "p"
            }, "need"), " to test your offline-first service worker locally, build\nthe application (using ", Object(a.b)("inlineCode", {
                parentName: "p"
            }, "npm run build"), ") and run a standard http server from your\nbuild directory. After running the build script, ", Object(a.b)("inlineCode", {
                parentName: "p"
            }, "create-react-app"), " will give\ninstructions for one way to test your production build locally and the ", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "/docs/deployment"
            }), "deployment instructions"), " have\ninstructions for using other methods. ", Object(a.b)("em", {
                parentName: "p"
            }, "Be sure to always use an\nincognito window to avoid complications with your browser cache."))), Object(a.b)("li", {
                parentName: "ol"
            }, Object(a.b)("p", {
                parentName: "li"
            }, "By default, the generated service worker file will not intercept or cache any\ncross-origin traffic, like HTTP ", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "/docs/integrating-with-an-api-backend"
            }), "API requests"), ",\nimages, or embeds loaded from a different domain."))), Object(a.b)("h2", {
                id: "progressive-web-app-metadata"
            }, "Progressive Web App Metadata"), Object(a.b)("p", null, "The default configuration includes a web app manifest located at\n", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://github.com/facebook/create-react-app/blob/master/packages/cra-template/template/public/manifest.json"
            }), Object(a.b)("inlineCode", {
                parentName: "a"
            }, "public/manifest.json")), ", that you can customize with\ndetails specific to your web application."), Object(a.b)("p", null, "When a user adds a web app to their homescreen using Chrome or Firefox on\nAndroid, the metadata in ", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://github.com/facebook/create-react-app/blob/master/packages/cra-template/template/public/manifest.json"
            }), Object(a.b)("inlineCode", {
                parentName: "a"
            }, "manifest.json")), " determines what\nicons, names, and branding colors to use when the web app is displayed.\n", Object(a.b)("a", Object(n.a)({
                parentName: "p"
            }, {
                href: "https://developers.google.com/web/fundamentals/engage-and-retain/web-app-manifest/"
            }), "The Web App Manifest guide"), "\nprovides more context about what each field means, and how your customizations\nwill affect your users' experience."), Object(a.b)("p", null, "Progressive web apps that have been added to the homescreen will load faster and\nwork offline when there's an active service worker. That being said, the\nmetadata from the web app manifest will still be used regardless of whether or\nnot you opt-in to service worker registration."))
        }
        u.isMDXComponent = !0
    },
    177: function(e, t, r) {
        "use strict";
        r.d(t, "a", (function() {
            return s
        }
        )),
        r.d(t, "b", (function() {
            return u
        }
        ));
        var n = r(0)
          , o = r.n(n)
          , a = o.a.createContext({})
          , i = function(e) {
            var t = o.a.useContext(a)
              , r = t;
            return e && (r = "function" == typeof e ? e(t) : Object.assign({}, t, e)),
            r
        }
          , s = function(e) {
            var t = i(e.components);
            return o.a.createElement(a.Provider, {
                value: t
            }, e.children)
        };
        var c = "mdxType"
          , l = {
            inlineCode: "code",
            wrapper: function(e) {
                var t = e.children;
                return o.a.createElement(o.a.Fragment, {}, t)
            }
        }
          , p = Object(n.forwardRef)((function(e, t) {
            var r = e.components
              , n = e.mdxType
              , a = e.originalType
              , s = e.parentName
              , c = function(e, t) {
                var r = {};
                for (var n in e)
                    Object.prototype.hasOwnProperty.call(e, n) && -1 === t.indexOf(n) && (r[n] = e[n]);
                return r
            }(e, ["components", "mdxType", "originalType", "parentName"])
              , p = i(r)
              , u = n
              , f = p[s + "." + u] || p[u] || l[u] || a;
            return r ? o.a.createElement(f, Object.assign({}, {
                ref: t
            }, c, {
                components: r
            })) : o.a.createElement(f, Object.assign({}, {
                ref: t
            }, c))
        }
        ));
        function u(e, t) {
            var r = arguments
              , n = t && t.mdxType;
            if ("string" == typeof e || n) {
                var a = r.length
                  , i = new Array(a);
                i[0] = p;
                var s = {};
                for (var l in t)
                    hasOwnProperty.call(t, l) && (s[l] = t[l]);
                s.originalType = e,
                s[c] = "string" == typeof e ? e : n,
                i[1] = s;
                for (var u = 2; u < a; u++)
                    i[u] = r[u];
                return o.a.createElement.apply(null, i)
            }
            return o.a.createElement.apply(null, r)
        }
        p.displayName = "MDXCreateElement"
    },
    178: function(e, t, r) {
        "use strict";
        e.exports = r(179)
    },
    179: function(e, t, r) {
        "use strict";
        var n = r(180)
          , o = "function" == typeof Symbol && Symbol.for
          , a = o ? Symbol.for("react.element") : 60103
          , i = o ? Symbol.for("react.portal") : 60106
          , s = o ? Symbol.for("react.fragment") : 60107
          , c = o ? Symbol.for("react.strict_mode") : 60108
          , l = o ? Symbol.for("react.profiler") : 60114
          , p = o ? Symbol.for("react.provider") : 60109
          , u = o ? Symbol.for("react.context") : 60110
          , f = o ? Symbol.for("react.forward_ref") : 60112
          , b = o ? Symbol.for("react.suspense") : 60113;
        o && Symbol.for("react.suspense_list");
        var d = o ? Symbol.for("react.memo") : 60115
          , h = o ? Symbol.for("react.lazy") : 60116;
        o && Symbol.for("react.fundamental"),
        o && Symbol.for("react.responder"),
        o && Symbol.for("react.scope");
        var m = "function" == typeof Symbol && Symbol.iterator;
        function y(e) {
            for (var t = "https://reactjs.org/docs/error-decoder.html?invariant=" + e, r = 1; r < arguments.length; r++)
                t += "&args[]=" + encodeURIComponent(arguments[r]);
            return "Minified React error #" + e + "; visit " + t + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings."
        }
        var g = {
            isMounted: function() {
                return !1
            },
            enqueueForceUpdate: function() {},
            enqueueReplaceState: function() {},
            enqueueSetState: function() {}
        }
          , w = {};
        function v(e, t, r) {
            this.props = e,
            this.context = t,
            this.refs = w,
            this.updater = r || g
        }
        function j() {}
        function O(e, t, r) {
            this.props = e,
            this.context = t,
            this.refs = w,
            this.updater = r || g
        }
        v.prototype.isReactComponent = {},
        v.prototype.setState = function(e, t) {
            if ("object" != typeof e && "function" != typeof e && null != e)
                throw Error(y(85));
            this.updater.enqueueSetState(this, e, t, "setState")
        }
        ,
        v.prototype.forceUpdate = function(e) {
            this.updater.enqueueForceUpdate(this, e, "forceUpdate")
        }
        ,
        j.prototype = v.prototype;
        var k = O.prototype = new j;
        k.constructor = O,
        n(k, v.prototype),
        k.isPureReactComponent = !0;
        var N = {
            current: null
        }
          , _ = {
            current: null
        }
          , C = Object.prototype.hasOwnProperty
          , S = {
            key: !0,
            ref: !0,
            __self: !0,
            __source: !0
        };
        function x(e, t, r) {
            var n, o = {}, i = null, s = null;
            if (null != t)
                for (n in void 0 !== t.ref && (s = t.ref),
                void 0 !== t.key && (i = "" + t.key),
                t)
                    C.call(t, n) && !S.hasOwnProperty(n) && (o[n] = t[n]);
            var c = arguments.length - 2;
            if (1 === c)
                o.children = r;
            else if (1 < c) {
                for (var l = Array(c), p = 0; p < c; p++)
                    l[p] = arguments[p + 2];
                o.children = l
            }
            if (e && e.defaultProps)
                for (n in c = e.defaultProps)
                    void 0 === o[n] && (o[n] = c[n]);
            return {
                $$typeof: a,
                type: e,
                key: i,
                ref: s,
                props: o,
                _owner: _.current
            }
        }
        function P(e) {
            return "object" == typeof e && null !== e && e.$$typeof === a
        }
        var T = /\/+/g
          , E = [];
        function A(e, t, r, n) {
            if (E.length) {
                var o = E.pop();
                return o.result = e,
                o.keyPrefix = t,
                o.func = r,
                o.context = n,
                o.count = 0,
                o
            }
            return {
                result: e,
                keyPrefix: t,
                func: r,
                context: n,
                count: 0
            }
        }
        function $(e) {
            e.result = null,
            e.keyPrefix = null,
            e.func = null,
            e.context = null,
            e.count = 0,
            10 > E.length && E.push(e)
        }
        function R(e, t, r) {
            return null == e ? 0 : function e(t, r, n, o) {
                var s = typeof t;
                "undefined" !== s && "boolean" !== s || (t = null);
                var c = !1;
                if (null === t)
                    c = !0;
                else
                    switch (s) {
                    case "string":
                    case "number":
                        c = !0;
                        break;
                    case "object":
                        switch (t.$$typeof) {
                        case a:
                        case i:
                            c = !0
                        }
                    }
                if (c)
                    return n(o, t, "" === r ? "." + I(t, 0) : r),
                    1;
                if (c = 0,
                r = "" === r ? "." : r + ":",
                Array.isArray(t))
                    for (var l = 0; l < t.length; l++) {
                        var p = r + I(s = t[l], l);
                        c += e(s, p, n, o)
                    }
                else if (null === t || "object" != typeof t ? p = null : p = "function" == typeof (p = m && t[m] || t["@@iterator"]) ? p : null,
                "function" == typeof p)
                    for (t = p.call(t),
                    l = 0; !(s = t.next()).done; )
                        c += e(s = s.value, p = r + I(s, l++), n, o);
                else if ("object" === s)
                    throw n = "" + t,
                    Error(y(31, "[object Object]" === n ? "object with keys {" + Object.keys(t).join(", ") + "}" : n, ""));
                return c
            }(e, "", t, r)
        }
        function I(e, t) {
            return "object" == typeof e && null !== e && null != e.key ? function(e) {
                var t = {
                    "=": "=0",
                    ":": "=2"
                };
                return "$" + ("" + e).replace(/[=:]/g, (function(e) {
                    return t[e]
                }
                ))
            }(e.key) : t.toString(36)
        }
        function W(e, t) {
            e.func.call(e.context, t, e.count++)
        }
        function M(e, t, r) {
            var n = e.result
              , o = e.keyPrefix;
            e = e.func.call(e.context, t, e.count++),
            Array.isArray(e) ? q(e, n, r, (function(e) {
                return e
            }
            )) : null != e && (P(e) && (e = function(e, t) {
                return {
                    $$typeof: a,
                    type: e.type,
                    key: t,
                    ref: e.ref,
                    props: e.props,
                    _owner: e._owner
                }
            }(e, o + (!e.key || t && t.key === e.key ? "" : ("" + e.key).replace(T, "$&/") + "/") + r)),
            n.push(e))
        }
        function q(e, t, r, n, o) {
            var a = "";
            null != r && (a = ("" + r).replace(T, "$&/") + "/"),
            R(e, M, t = A(t, a, n, o)),
            $(t)
        }
        function U() {
            var e = N.current;
            if (null === e)
                throw Error(y(321));
            return e
        }
        var F = {
            Children: {
                map: function(e, t, r) {
                    if (null == e)
                        return e;
                    var n = [];
                    return q(e, n, null, t, r),
                    n
                },
                forEach: function(e, t, r) {
                    if (null == e)
                        return e;
                    R(e, W, t = A(null, null, t, r)),
                    $(t)
                },
                count: function(e) {
                    return R(e, (function() {
                        return null
                    }
                    ), null)
                },
                toArray: function(e) {
                    var t = [];
                    return q(e, t, null, (function(e) {
                        return e
                    }
                    )),
                    t
                },
                only: function(e) {
                    if (!P(e))
                        throw Error(y(143));
                    return e
                }
            },
            createRef: function() {
                return {
                    current: null
                }
            },
            Component: v,
            PureComponent: O,
            createContext: function(e, t) {
                return void 0 === t && (t = null),
                (e = {
                    $$typeof: u,
                    _calculateChangedBits: t,
                    _currentValue: e,
                    _currentValue2: e,
                    _threadCount: 0,
                    Provider: null,
                    Consumer: null
                }).Provider = {
                    $$typeof: p,
                    _context: e
                },
                e.Consumer = e
            },
            forwardRef: function(e) {
                return {
                    $$typeof: f,
                    render: e
                }
            },
            lazy: function(e) {
                return {
                    $$typeof: h,
                    _ctor: e,
                    _status: -1,
                    _result: null
                }
            },
            memo: function(e, t) {
                return {
                    $$typeof: d,
                    type: e,
                    compare: void 0 === t ? null : t
                }
            },
            useCallback: function(e, t) {
                return U().useCallback(e, t)
            },
            useContext: function(e, t) {
                return U().useContext(e, t)
            },
            useEffect: function(e, t) {
                return U().useEffect(e, t)
            },
            useImperativeHandle: function(e, t, r) {
                return U().useImperativeHandle(e, t, r)
            },
            useDebugValue: function() {},
            useLayoutEffect: function(e, t) {
                return U().useLayoutEffect(e, t)
            },
            useMemo: function(e, t) {
                return U().useMemo(e, t)
            },
            useReducer: function(e, t, r) {
                return U().useReducer(e, t, r)
            },
            useRef: function(e) {
                return U().useRef(e)
            },
            useState: function(e) {
                return U().useState(e)
            },
            Fragment: s,
            Profiler: l,
            StrictMode: c,
            Suspense: b,
            createElement: x,
            cloneElement: function(e, t, r) {
                if (null == e)
                    throw Error(y(267, e));
                var o = n({}, e.props)
                  , i = e.key
                  , s = e.ref
                  , c = e._owner;
                if (null != t) {
                    if (void 0 !== t.ref && (s = t.ref,
                    c = _.current),
                    void 0 !== t.key && (i = "" + t.key),
                    e.type && e.type.defaultProps)
                        var l = e.type.defaultProps;
                    for (p in t)
                        C.call(t, p) && !S.hasOwnProperty(p) && (o[p] = void 0 === t[p] && void 0 !== l ? l[p] : t[p])
                }
                var p = arguments.length - 2;
                if (1 === p)
                    o.children = r;
                else if (1 < p) {
                    l = Array(p);
                    for (var u = 0; u < p; u++)
                        l[u] = arguments[u + 2];
                    o.children = l
                }
                return {
                    $$typeof: a,
                    type: e.type,
                    key: i,
                    ref: s,
                    props: o,
                    _owner: c
                }
            },
            createFactory: function(e) {
                var t = x.bind(null, e);
                return t.type = e,
                t
            },
            isValidElement: P,
            version: "16.11.0",
            __SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED: {
                ReactCurrentDispatcher: N,
                ReactCurrentBatchConfig: {
                    suspense: null
                },
                ReactCurrentOwner: _,
                IsSomeRendererActing: {
                    current: !1
                },
                assign: n
            }
        }
          , B = {
            default: F
        }
          , L = B && F || B;
        e.exports = L.default || L
    },
    180: function(e, t, r) {
        "use strict";
        var n = Object.getOwnPropertySymbols
          , o = Object.prototype.hasOwnProperty
          , a = Object.prototype.propertyIsEnumerable;
        function i(e) {
            if (null == e)
                throw new TypeError("Object.assign cannot be called with null or undefined");
            return Object(e)
        }
        e.exports = function() {
            try {
                if (!Object.assign)
                    return !1;
                var e = new String("abc");
                if (e[5] = "de",
                "5" === Object.getOwnPropertyNames(e)[0])
                    return !1;
                for (var t = {}, r = 0; r < 10; r++)
                    t["_" + String.fromCharCode(r)] = r;
                if ("0123456789" !== Object.getOwnPropertyNames(t).map((function(e) {
                    return t[e]
                }
                )).join(""))
                    return !1;
                var n = {};
                return "abcdefghijklmnopqrst".split("").forEach((function(e) {
                    n[e] = e
                }
                )),
                "abcdefghijklmnopqrst" === Object.keys(Object.assign({}, n)).join("")
            } catch (o) {
                return !1
            }
        }() ? Object.assign : function(e, t) {
            for (var r, s, c = i(e), l = 1; l < arguments.length; l++) {
                for (var p in r = Object(arguments[l]))
                    o.call(r, p) && (c[p] = r[p]);
                if (n) {
                    s = n(r);
                    for (var u = 0; u < s.length; u++)
                        a.call(r, s[u]) && (c[s[u]] = r[s[u]])
                }
            }
            return c
        }
    }
}]);
