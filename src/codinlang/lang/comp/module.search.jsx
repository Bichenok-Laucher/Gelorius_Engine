module myApp.Search {
    'use strict';

    export class SearchResult {
        id: string;
        title: string;

        constructor(result, term?: string) {
            this.id = result.id;
            this.title = term ? Utils.highlightSearchTerm(term, result.title) : result.title;
        }
    }
}