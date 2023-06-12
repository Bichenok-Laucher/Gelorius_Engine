module myApp.Search {
    'use strict';

    export class Utils {
        private static regexFlags = 'gi';
        private static wrapper = 'mark';

        private static wrap(match: string): string {
            return '<' + Utils.wrapper + '>' + match + '</' + Utils.wrapper + '>';
        }

        static highlightSearchTerm(term: string, searchResult: string): string {
            let regex = new RegExp(term, Utils.regexFlags);

            return searchResult.replace(regex, match => Utils.wrap(match));
        }
    }
}