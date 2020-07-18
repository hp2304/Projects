package com.resumeBuilder.controller.admin;

import com.resumeBuilder.view.admin.mngCompany.AdmCompDtlsDsply;
import com.resumeBuilder.view.admin.mngCompany.AdmMngCompanyView;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class CompDtlsViewListener implements ActionListener {

    private AdmCompDtlsDsply compDtlsView;

    public CompDtlsViewListener(AdmCompDtlsDsply compDtlsView) {
        this.compDtlsView = compDtlsView;

    }

    @Override
    public void actionPerformed(ActionEvent e) {
        try {
            AdmMngCompanyView frame = new AdmMngCompanyView();
            frame.setVisible(true);
            compDtlsView.setVisible(false);
        } catch (Exception error) {
            error.printStackTrace();
        }

    }


}
