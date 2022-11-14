"use strict";(self.webpackChunkdevelopers=self.webpackChunkdevelopers||[]).push([[554],{3905:(t,e,a)=>{a.d(e,{Zo:()=>d,kt:()=>m});var r=a(7294);function n(t,e,a){return e in t?Object.defineProperty(t,e,{value:a,enumerable:!0,configurable:!0,writable:!0}):t[e]=a,t}function l(t,e){var a=Object.keys(t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(t);e&&(r=r.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),a.push.apply(a,r)}return a}function o(t){for(var e=1;e<arguments.length;e++){var a=null!=arguments[e]?arguments[e]:{};e%2?l(Object(a),!0).forEach((function(e){n(t,e,a[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(a)):l(Object(a)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(a,e))}))}return t}function i(t,e){if(null==t)return{};var a,r,n=function(t,e){if(null==t)return{};var a,r,n={},l=Object.keys(t);for(r=0;r<l.length;r++)a=l[r],e.indexOf(a)>=0||(n[a]=t[a]);return n}(t,e);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(t);for(r=0;r<l.length;r++)a=l[r],e.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(t,a)&&(n[a]=t[a])}return n}var s=r.createContext({}),p=function(t){var e=r.useContext(s),a=e;return t&&(a="function"==typeof t?t(e):o(o({},e),t)),a},d=function(t){var e=p(t.components);return r.createElement(s.Provider,{value:e},t.children)},c={inlineCode:"code",wrapper:function(t){var e=t.children;return r.createElement(r.Fragment,{},e)}},u=r.forwardRef((function(t,e){var a=t.components,n=t.mdxType,l=t.originalType,s=t.parentName,d=i(t,["components","mdxType","originalType","parentName"]),u=p(a),m=n,h=u["".concat(s,".").concat(m)]||u[m]||c[m]||l;return a?r.createElement(h,o(o({ref:e},d),{},{components:a})):r.createElement(h,o({ref:e},d))}));function m(t,e){var a=arguments,n=e&&e.mdxType;if("string"==typeof t||n){var l=a.length,o=new Array(l);o[0]=u;var i={};for(var s in e)hasOwnProperty.call(e,s)&&(i[s]=e[s]);i.originalType=t,i.mdxType="string"==typeof t?t:n,o[1]=i;for(var p=2;p<l;p++)o[p]=a[p];return r.createElement.apply(null,o)}return r.createElement.apply(null,a)}u.displayName="MDXCreateElement"},2055:(t,e,a)=>{a.r(e),a.d(e,{assets:()=>s,contentTitle:()=>o,default:()=>c,frontMatter:()=>l,metadata:()=>i,toc:()=>p});var r=a(7462),n=(a(7294),a(3905));a(8209);const l={title:"Datasets",sidebar_position:5},o=void 0,i={unversionedId:"datasets",id:"datasets",title:"Datasets",description:"We have two datasets available for download for use in your research.",source:"@site/docs/datasets.md",sourceDirName:".",slug:"/datasets",permalink:"/docs/datasets",draft:!1,tags:[],version:"current",sidebarPosition:5,frontMatter:{title:"Datasets",sidebar_position:5},sidebar:"tutorialSidebar",previous:{title:"IIIF",permalink:"/docs/iiif"},next:{title:"API Reference",permalink:"/docs/api"}},s={},p=[{value:"Catalogue snapshot",id:"catalogue-snapshot",level:2},{value:"London MOH reports",id:"london-moh-reports",level:2}],d={toc:p};function c(t){let{components:e,...a}=t;return(0,n.kt)("wrapper",(0,r.Z)({},d,a,{components:e,mdxType:"MDXLayout"}),(0,n.kt)("p",null,"We have two datasets available for download for use in your research."),(0,n.kt)("h2",{id:"catalogue-snapshot"},"Catalogue snapshot"),(0,n.kt)("p",null,"This dataset provides a daily snapshot of the catalogue that describes our museum and library collections. Downloads are line-delimited JSON, with each line providing one resource in the same serialisation format as the ",(0,n.kt)("a",{parentName:"p",href:"/docs/catalogue"},"Catalogue API"),"."),(0,n.kt)("table",null,(0,n.kt)("thead",{parentName:"table"},(0,n.kt)("tr",{parentName:"thead"},(0,n.kt)("th",{parentName:"tr",align:null},"Description"),(0,n.kt)("th",{parentName:"tr",align:null},"Size"),(0,n.kt)("th",{parentName:"tr",align:null},"Download"))),(0,n.kt)("tbody",{parentName:"table"},(0,n.kt)("tr",{parentName:"tbody"},(0,n.kt)("td",{parentName:"tr",align:null},"All works as JSON"),(0,n.kt)("td",{parentName:"tr",align:null},"1.3 GB"),(0,n.kt)("td",{parentName:"tr",align:null},(0,n.kt)("a",{parentName:"td",href:"https://data.wellcomecollection.org/catalogue/v2/works.json.gz"},"works.json.gz"))))),(0,n.kt)("h2",{id:"london-moh-reports"},"London MOH reports"),(0,n.kt)("p",null,"This dataset brings together around 5800 Medical Officer of Health (MOH) reports from the Greater London area. This includes the present-day City of London, 32 London boroughs and the predecessor local authorities for these boroughs, including urban and rural district councils and sanitary districts. Full text of the reports is included, along with around 275,000 tables that have been extracted as individual files. The extracted tables have undergone extensive quality assurance checks, but due to the volume of the data, we cannot promise 100% accuracy. The data is licensed under a ",(0,n.kt)("a",{parentName:"p",href:"https://creativecommons.org/licenses/by/4.0/"},"Creative Commons Attribution 4.0 International Licence"),"."),(0,n.kt)("table",null,(0,n.kt)("thead",{parentName:"table"},(0,n.kt)("tr",{parentName:"thead"},(0,n.kt)("th",{parentName:"tr",align:null},"Description"),(0,n.kt)("th",{parentName:"tr",align:null},"Size"),(0,n.kt)("th",{parentName:"tr",align:null},"Download"))),(0,n.kt)("tbody",{parentName:"table"},(0,n.kt)("tr",{parentName:"tbody"},(0,n.kt)("td",{parentName:"tr",align:null},"Full text corpus (raw text)"),(0,n.kt)("td",{parentName:"tr",align:null},"215 MB"),(0,n.kt)("td",{parentName:"tr",align:null},(0,n.kt)("a",{parentName:"td",href:"https://data.wellcomecollection.org/moh/Fulltext.zip"},"Fulltext.zip"))),(0,n.kt)("tr",{parentName:"tbody"},(0,n.kt)("td",{parentName:"tr",align:null},"All report tables as CSV"),(0,n.kt)("td",{parentName:"tr",align:null},"340 MB"),(0,n.kt)("td",{parentName:"tr",align:null},(0,n.kt)("a",{parentName:"td",href:"https://data.wellcomecollection.org/moh/All_report_tables.csv.zip"},"All_report_tables.csv.zip"))),(0,n.kt)("tr",{parentName:"tbody"},(0,n.kt)("td",{parentName:"tr",align:null},"All report tables as HTML"),(0,n.kt)("td",{parentName:"tr",align:null},"412 MB"),(0,n.kt)("td",{parentName:"tr",align:null},(0,n.kt)("a",{parentName:"td",href:"https://data.wellcomecollection.org/moh/All_report_tables.html.zip"},"All_report_tables.html.zip"))),(0,n.kt)("tr",{parentName:"tbody"},(0,n.kt)("td",{parentName:"tr",align:null},"All report tables as XML"),(0,n.kt)("td",{parentName:"tr",align:null},"536 MB"),(0,n.kt)("td",{parentName:"tr",align:null},(0,n.kt)("a",{parentName:"td",href:"https://data.wellcomecollection.org/moh/All_report_tables.xml.zip"},"All_report_tables.xml.zip"))),(0,n.kt)("tr",{parentName:"tbody"},(0,n.kt)("td",{parentName:"tr",align:null},"All report tables as TXT"),(0,n.kt)("td",{parentName:"tr",align:null},"422 MB"),(0,n.kt)("td",{parentName:"tr",align:null},(0,n.kt)("a",{parentName:"td",href:"https://data.wellcomecollection.org/moh/All_report_tables.txt.zip"},"All_report_tables.txt.zip"))))))}c.isMDXComponent=!0},8209:(t,e,a)=>{a(7294)}}]);